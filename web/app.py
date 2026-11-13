"""
app.py - FastAPI web application for PropTerminal NBA prop intelligence

Routes:
- GET /              -> Dashboard (today's games, top picks)
- GET /game/{id}     -> Game detail (all projections)
- GET /player/{name} -> Player profile
- GET /api/games     -> JSON today's games
- GET /api/game/{id}/analyze -> JSON GameRecommendations
- GET /api/player/{name}     -> JSON PlayerProfile
- POST /api/refresh  -> Clear caches
- GET /login         -> Login page
- POST /login        -> Login submit
- GET /logout        -> Logout (clear session)
"""

import hashlib
import json
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse as _OrigJSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class JSONResponse(_OrigJSONResponse):
    def render(self, content):
        return json.dumps(content, cls=_NumpySafeEncoder).encode("utf-8")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PropTerminal", description="Real-time NBA Prop Intelligence")

# Middleware stack: added in reverse order (last added = first to run)
# Auth middleware runs AFTER session middleware has populated request.session
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in ("/login", "/health") or path.startswith("/static"):
            return await call_next(request)
        if path == "/login" and request.method == "POST":
            return await call_next(request)
        if not request.session.get("authenticated"):
            return RedirectResponse(url="/login", status_code=302)
        return await call_next(request)

app.add_middleware(AuthMiddleware)  # Runs second (after session is available)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "propterminal-dev-secret-change-me"),
)  # Runs first (populates request.session)

# Auth config: set PT_PASSWORD_HASH env var to SHA-256 of your password
# Default password for dev: "propterminal"
AUTH_PASSWORD_HASH = os.environ.get("PT_PASSWORD_HASH", "")
if not AUTH_PASSWORD_HASH:
    AUTH_PASSWORD_HASH = hashlib.sha256("propterminal".encode()).hexdigest()

# Mount static files and templates
web_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(web_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(web_dir, "templates"))

# Lazy engine initialization (expensive - loads ML models)
_engine = None  # type: Optional[object]


def get_engine():
    """Lazy-load the analysis engine singleton."""
    global _engine
    if _engine is None:
        from web.engine import AnalysisEngine
        _engine = AnalysisEngine.get_instance()
    return _engine


def _ml_filter_status():
    # type: () -> str
    """Return which ML model version is active."""
    try:
        from pathlib import Path
        import json
        import os
        models_dir = Path(os.environ.get("AIBALL_MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models" / "v3_classifier_holdout")))
        meta_path = models_dir / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_type = meta.get("model_type", "")
            if model_type == "classifier":
                return "v3 Classifier"
            return "v2 Model"
    except Exception:
        pass

    try:
        from ml_filter import MLFilter
        f = MLFilter()
        if f.loaded:
            # Check for v10 models first
            if f.prop_models and any('v10' in str(getattr(m, '_model_path', '')) for m in f.prop_models.values()):
                return "V10 ML Filter"
            if hasattr(f, 'prop_metadata') and f.prop_metadata:
                version = f.prop_metadata.get('version', '')
                if version:
                    return f"{version} ML Filter"
            if f.prop_models:
                return "V10 ML Filter"
            return "Legacy ML Filter"
    except Exception:
        pass
    return "Rule-based"


# =========================================================================
# AUTH MIDDLEWARE & ROUTES
# =========================================================================

    # Auth middleware is now a class-based middleware (AuthMiddleware above)
    # to ensure it runs AFTER SessionMiddleware populates request.session


@app.get("/health")
async def health():
    """Health check for ALB/container orchestration."""
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render login page."""
    # If already authenticated, redirect to dashboard
    if request.session.get("authenticated"):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@app.post("/login")
async def login_submit(request: Request):
    """Handle login form submission."""
    form = await request.form()
    password = form.get("password", "")
    if hashlib.sha256(password.encode()).hexdigest() == AUTH_PASSWORD_HASH:
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": "Wrong password"})


@app.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


# =========================================================================
# HTML ROUTES
# =========================================================================


@app.on_event("startup")
async def startup_preload():
    """Auto-analyze today's games in background on server start."""
    async def _preload():
        try:
            await asyncio.sleep(3)  # Let server finish starting
            engine = get_engine()
            games = await asyncio.to_thread(engine.get_games_for_date)
            logger.info(f"Pre-loading analysis for {len(games)} games...")
            for g in games:
                gid = g["game_id"]
                try:
                    await asyncio.to_thread(engine.analyze_game, gid)
                    logger.info(f"Pre-loaded game {gid}")
                except Exception as e:
                    logger.debug(f"Pre-load failed for {gid}: {e}")
            logger.info("Pre-load complete — all games cached")
        except Exception as e:
            logger.debug(f"Pre-load error: {e}")
    asyncio.create_task(_preload())
    # Start the daily auto-analyze scheduler
    asyncio.create_task(_daily_analyze_scheduler())
    # Start the CLV closing line capture scheduler
    asyncio.create_task(_clv_closing_line_scheduler())
    # Paper log of the unders screen (no bets are placed). Set PAPER_LOG_ENABLED=0 to switch it off.
    if os.environ.get("PAPER_LOG_ENABLED", "1") == "1":
        asyncio.create_task(_paper_log_scheduler())


async def _paper_log_scheduler():
    """Every 30 minutes from 9:00 to 21:30 Eastern while the server is up: pull live prices (Odds API, about 5 credits a game)
    and log unders priced 2%+ above the fair price (paper_log.py). Once a day: refresh box scores and grade finished games."""
    import pytz
    et = pytz.timezone("US/Eastern")
    await asyncio.sleep(90)
    graded_on = None
    logger.info("Paper log scheduler started")
    while True:
        try:
            import paper_log
            now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(et)
            if graded_on != now_et.date() and now_et.hour >= 8:
                out = await asyncio.to_thread(paper_log.scheduled, "grade")
                graded_on = now_et.date()
                logger.info("Paper log graded: %s", " | ".join(out.strip().splitlines()[-6:]))
            if 9 <= now_et.hour <= 21:
                out = await asyncio.to_thread(paper_log.scheduled, "poll")
                logger.info("Paper log poll: %s", " | ".join(out.strip().splitlines()[:2]))
        except Exception as e:
            logger.error("Paper log scheduler error: %s", e)
        now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(et)
        await asyncio.sleep(max(60, (30 - now_et.minute % 30) * 60 - now_et.second + 5))


async def _daily_analyze_scheduler():
    """Run full analysis daily at 3pm ET (20:00 UTC during EST, 19:00 UTC during EDT)."""
    import pytz
    et = pytz.timezone("US/Eastern")

    while True:
        try:
            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
            now_et = now_utc.astimezone(et)

            # Next 3pm ET
            target = now_et.replace(hour=15, minute=0, second=0, microsecond=0)
            if now_et >= target:
                target += timedelta(days=1)

            wait_seconds = (target - now_et).total_seconds()
            logger.info(f"Daily auto-analyze scheduled for {target.strftime('%Y-%m-%d %I:%M %p %Z')} "
                        f"({wait_seconds/3600:.1f}h from now)")
            await asyncio.sleep(wait_seconds)

            # Run analysis
            logger.info("Daily auto-analyze starting...")
            engine = get_engine()
            # Clear cache so we get fresh lines
            engine._analysis_cache = {}
            games = await asyncio.to_thread(engine.get_games_for_date)
            analyzed = 0
            for g in games:
                gid = g["game_id"]
                try:
                    await asyncio.to_thread(engine.analyze_game, gid)
                    analyzed += 1
                    logger.info(f"Auto-analyzed game {gid}")
                except Exception as e:
                    logger.error(f"Auto-analyze failed for {gid}: {e}")
            logger.info(f"Daily auto-analyze complete: {analyzed}/{len(games)} games")

            # Sleep a bit to avoid re-triggering immediately
            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Daily scheduler error: {e}")
            await asyncio.sleep(300)  # Retry in 5 min on error


async def _clv_closing_line_scheduler():
    """
    Background job that captures closing lines and computes CLV.

    Runs every 5 minutes:
    1. For games about to tip off (within 5 min), capture closing lines
    2. For finished games, compute CLV with actual results
    """
    # Wait for initial load to complete
    await asyncio.sleep(30)
    logger.info("CLV closing line scheduler started")

    while True:
        try:
            engine = get_engine()
            tracker = engine.clv_tracker

            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            # Phase 1: Capture closing lines for games transitioning to live
            # (This is also handled in get_live_predictions, but this is a safety net)
            games_needing_close = tracker.get_games_needing_closing_lines()
            for game_info in games_needing_close:
                gid = game_info["game_id"]
                if gid in engine._closing_lines_captured:
                    continue
                # Check if this game has started (use prediction cache as proxy)
                if gid in engine._prediction_cache:
                    # Build fresh props from current PropLine data
                    try:
                        if engine.use_propline:
                            current_props = engine.propline.get_all_props(
                                bookmaker="draftkings", date_str=game_info.get("game_date", today)
                            )
                            fresh_by_key = {}
                            for event_id, props in current_props.items():
                                for prop in props:
                                    pname = prop.get("player_name", "")
                                    ptype = prop.get("prop_type", "")
                                    if pname and ptype:
                                        pkey = "{}|{}".format(pname, ptype)
                                        if pkey in engine._prediction_cache.get(gid, {}):
                                            fresh_by_key[pkey] = {
                                                "line": prop.get("line", 0),
                                                "over_odds": prop.get("over_odds", -110),
                                                "under_odds": prop.get("under_odds", -110),
                                            }
                            if fresh_by_key:
                                engine.capture_closing_lines(
                                    gid, fresh_by_key, game_info.get("game_date")
                                )
                    except Exception as e:
                        logger.debug("CLV scheduler: closing line capture failed for %d: %s", gid, e)

            # Phase 2: Compute CLV for finished games
            # Check yesterday's and today's games for final status
            for date_str in [yesterday, today]:
                try:
                    games = await asyncio.to_thread(engine.get_games_for_date, date_str)
                    for g in games:
                        if g.get("status_category") != "final":
                            continue
                        gid = g["game_id"]
                        # Try to get actual results and compute CLV
                        try:
                            game_results = await asyncio.to_thread(engine.get_game_results, gid)
                            if game_results and game_results.get("results"):
                                actual_results = {}
                                for r in game_results["results"]:
                                    pkey = "{}|{}".format(r["player"], r["prop_type"])
                                    actual_results[pkey] = {
                                        "actual": r.get("actual", 0),
                                        "hit": r.get("hit", False),
                                    }
                                await asyncio.to_thread(
                                    tracker.compute_clv_for_game, gid, actual_results
                                )
                        except Exception as e:
                            logger.debug("CLV compute for game %d: %s", gid, e)
                except Exception as e:
                    logger.debug("CLV scheduler date %s: %s", date_str, e)

            await asyncio.sleep(300)  # Run every 5 minutes

        except Exception as e:
            logger.error("CLV scheduler error: %s", e)
            await asyncio.sleep(300)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing today's games."""
    import time as _t
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "ml_status": _ml_filter_status(),
        "cache_bust": str(int(_t.time())),
    })


@app.get("/game/{game_id}", response_class=HTMLResponse)
async def game_detail(request: Request, game_id: int):
    """Game detail page with all projections."""
    import time as _t
    return templates.TemplateResponse(request=request, name="game.html", context={
        "game_id": game_id,
        "cache_bust": str(int(_t.time())),
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "ml_status": _ml_filter_status(),
    })


@app.get("/player/{player_name}", response_class=HTMLResponse)
async def player_detail(request: Request, player_name: str):
    """Player profile page."""
    return templates.TemplateResponse(request=request, name="player.html", context={
        "player_name": player_name,
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "ml_status": _ml_filter_status(),
    })


# =========================================================================
# API ROUTES
# =========================================================================

@app.get("/api/live-lines")
async def api_live_lines(date: Optional[str] = None):
    """
    Fast endpoint that returns current lines for all games.
    Called every 30-60 seconds by the frontend.
    Uses PropLine bulk endpoint (single API call, <1s).
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_live_lines, date)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error fetching live lines: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/live-predictions")
async def api_live_predictions(date: Optional[str] = None):
    """
    Real-time predictions endpoint. Recalculates edge for all cached
    predictions using fresh odds from PropLine. Called every 30s.

    Returns predictions with current edge, kelly, line shopping data.
    Falls back gracefully if no games analyzed yet.
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_live_predictions, date)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error in live predictions: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/api/games")
async def api_games(date: Optional[str] = None):
    """Get games for a date with spreads, totals, injury counts. Defaults to today."""
    try:
        engine = get_engine()
        games = await asyncio.to_thread(engine.get_games_for_date, date)
        selected_date = date or datetime.now().strftime("%Y-%m-%d")
        return JSONResponse(content={"games": games, "count": len(games), "date": selected_date})
    except Exception as e:
        logger.error("Error fetching games: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e), "games": [], "count": 0},
            status_code=500
        )


@app.get("/api/game/{game_id}/analyze")
async def api_analyze_game(game_id: int):
    """
    Run full analysis on a game. This is the heavy call (15-30s).

    Returns GameRecommendations with singles, parlays, projections.
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.analyze_game, game_id)
        if result is None:
            return JSONResponse(
                content={"error": "Game not found or no data available"},
                status_code=404
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error analyzing game %d: %s", game_id, str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/player/{player_name}")
async def api_player(player_name: str):
    """Get player profile with stats, splits, game logs."""
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_player_profile, player_name)
        if result is None:
            return JSONResponse(
                content={"error": "Player not found: {}".format(player_name)},
                status_code=404
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error fetching player %s: %s", player_name, str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# =========================================================================
# INJURY CASCADE API
# =========================================================================

@app.get("/api/injury-cascade")
async def api_injury_cascade():
    """
    Get current injury cascade alerts.

    Returns pre-game cascade scan results and any real-time alerts
    from injury status changes.
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_injury_cascade_alerts)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error fetching cascade alerts: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"alerts": [], "pregame_cascades": [], "status": "error", "message": str(e)},
            status_code=500
        )


@app.get("/api/injury-cascade/lookup")
async def api_injury_cascade_lookup(player: str):
    """
    Look up cascade impacts if a specific player sits out.

    Query param: ?player=Jaylen Brown
    Returns all teammate stat changes based on historical data.
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.lookup_injury_cascade, player)
        if "error" in result:
            return JSONResponse(content=result, status_code=503)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error in cascade lookup for %s: %s", player, str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/game/{game_id}/results")
async def api_game_results(game_id: int):
    """Get graded results for a finished game via PropLine."""
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_game_results, game_id)
        if result is None:
            return JSONResponse(
                content={"error": "No results available for this game"},
                status_code=404
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error fetching results for game %d: %s", game_id, str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/api/analyze-all")
async def api_analyze_all(date: Optional[str] = None):
    """
    Kick off analysis for ALL games on a date. Runs in background threads.
    Returns immediately with game IDs being analyzed.
    Results are cached — poll /api/game/{id}/analyze for each.
    """
    try:
        engine = get_engine()
        games = await asyncio.to_thread(engine.get_games_for_date, date)
        game_ids = [g["game_id"] for g in games]

        # Analyze games SEQUENTIALLY in background to avoid API rate limit overload
        async def _analyze_all_sequential(gids):
            for gid in gids:
                cached = gid in engine._analysis_cache
                if not cached or (isinstance(engine._analysis_cache[gid], tuple) and
                    __import__('time').time() - engine._analysis_cache[gid][0] > engine._cache_ttl):
                    try:
                        await asyncio.to_thread(engine.analyze_game, gid)
                        logger.info(f"Background analysis complete for game {gid}")
                    except Exception as e:
                        logger.error(f"Background analysis failed for game {gid}: {e}")

        asyncio.create_task(_analyze_all_sequential(game_ids))

        already_cached = sum(1 for gid in game_ids if gid in engine._analysis_cache)
        return JSONResponse(content={
            "status": "ok",
            "games_queued": len(game_ids) - already_cached,
            "games_cached": already_cached,
            "game_ids": game_ids,
            "message": f"Analyzing {len(game_ids) - already_cached} games in background"
        })
    except Exception as e:
        logger.error("Error starting bulk analysis: %s", str(e), exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/game/{game_id}/reanalyze")
async def api_reanalyze_game(game_id: int):
    """
    Clear a single game's cache and re-run analysis.
    Called when live polling detects a line has moved from the analyzed value.
    """
    try:
        engine = get_engine()
        # Remove this game from the analysis cache
        if game_id in engine._analysis_cache:
            del engine._analysis_cache[game_id]
            logger.info("Cleared cache for game %d — re-analyzing", game_id)
        result = await asyncio.to_thread(engine.analyze_game, game_id)
        if result is None:
            return JSONResponse(
                content={"error": "Game not found or no data available"},
                status_code=404,
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error re-analyzing game %d: %s", game_id, str(e), exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/paper-log")
async def api_paper_log(limit: int = 50):
    """Paper log of the unders screen (no bets placed): the report by rule, and the latest flags."""
    try:
        import paper_log
        report = await asyncio.to_thread(paper_log.scheduled, "report")
        recent = await asyncio.to_thread(paper_log.recent_flags, max(1, min(limit, 500)))
        return JSONResponse({"report": report, "recent_flags": recent})
    except Exception as e:
        logger.error("Paper log report failed: %s", e)
        return JSONResponse({"error": "paper log unavailable"}, status_code=500)


@app.get("/api/stale-lines")
async def api_stale_lines(date: Optional[str] = None):
    """
    Get all stale lines detected for a date's games.
    Returns props where DK line appears stale (not updated despite new information).
    """
    try:
        engine = get_engine()
        # Get cached analysis results, or analyze if needed
        games = await asyncio.to_thread(engine.get_games_for_date, date)
        all_stale = []
        for g in games:
            gid = g["game_id"]
            # Use cached analysis if available
            import time as _time
            if gid in engine._analysis_cache:
                cached_at, cached_result = engine._analysis_cache[gid]
                if _time.time() - cached_at < engine._cache_ttl:
                    stale_data = cached_result.get("stale_lines", {})
                    stale_items = stale_data.get("stale_lines", [])
                    for item in stale_items:
                        item["game_id"] = gid
                        item["matchup"] = cached_result.get("matchup", "")
                    all_stale.extend(stale_items)

        # Sort by staleness score descending
        all_stale.sort(key=lambda x: x.get("staleness_score", 0), reverse=True)

        return JSONResponse(content={
            "stale_lines": all_stale,
            "count": len(all_stale),
            "date": date or datetime.now().strftime("%Y-%m-%d"),
        })
    except Exception as e:
        logger.error("Error fetching stale lines: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e), "stale_lines": [], "count": 0},
            status_code=500,
        )


@app.post("/api/refresh")
async def api_refresh():
    """Clear all caches and force fresh data."""
    try:
        engine = get_engine()
        await asyncio.to_thread(engine.clear_caches)
        return JSONResponse(content={"status": "ok", "message": "Caches cleared"})
    except Exception as e:
        logger.error("Error clearing caches: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/best-lines")
async def api_best_lines(date: Optional[str] = None):
    """
    Best available lines across all books for today's games.

    Returns per-prop best over/under book, consensus line, and DK reference.
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_best_lines_for_date, date)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error fetching best lines: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/shopping-summary")
async def api_shopping_summary():
    """
    Aggregate line shopping stats from current session's analyzed games.

    Returns EV improvement, book frequency, arbitrage/middle counts.
    """
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.get_shopping_summary)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error computing shopping summary: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# =========================================================================
# CLV TRACKING ENDPOINTS
# =========================================================================

@app.get("/api/clv-report")
async def api_clv_report(date: Optional[str] = None, days: Optional[int] = None):
    """
    CLV report for a specific date or time window.

    Query params:
        date: YYYY-MM-DD for a single date
        days: number of days to look back (7, 30, etc.)
        (neither: all-time)
    """
    try:
        engine = get_engine()
        tracker = engine.clv_tracker

        def _get_report():
            if date:
                return tracker.get_clv_report(date_str=date)
            elif days:
                return tracker.get_clv_report(days=days)
            else:
                return tracker.get_clv_report()

        result = await asyncio.to_thread(_get_report)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error getting CLV report: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/clv-summary")
async def api_clv_summary():
    """
    CLV summary across multiple time windows (7d, 30d, all-time).

    Returns {last_7d: {...}, last_30d: {...}, all_time: {...}}
    """
    try:
        engine = get_engine()
        tracker = engine.clv_tracker
        result = await asyncio.to_thread(tracker.get_clv_summary)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error getting CLV summary: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/api/clv-tracking-stats")
async def api_clv_tracking_stats():
    """
    CLV tracking system statistics (how much data we have).

    Returns snapshot/recommendation/result counts and date ranges.
    """
    try:
        engine = get_engine()
        tracker = engine.clv_tracker
        result = await asyncio.to_thread(tracker.get_tracking_stats)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error getting CLV tracking stats: %s", str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/api/clv-compute/{game_id}")
async def api_clv_compute(game_id: int):
    """
    Manually trigger CLV computation for a specific game.

    Useful after a game is final and closing lines + results are available.
    """
    try:
        engine = get_engine()

        def _compute():
            # Try to get actual results from PropLine
            actual_results = None
            try:
                game_results = engine.get_game_results(game_id)
                if game_results and game_results.get("results"):
                    actual_results = {}
                    for r in game_results["results"]:
                        pkey = "{}|{}".format(r["player"], r["prop_type"])
                        actual_results[pkey] = {
                            "actual": r.get("actual", 0),
                            "hit": r.get("hit", False),
                        }
            except Exception as e:
                logger.debug("Could not fetch results for CLV computation: %s", e)

            return engine.compute_clv_for_game(game_id, actual_results)

        results = await asyncio.to_thread(_compute)
        if results is None:
            return JSONResponse(
                content={"error": "No recommendations found for this game"},
                status_code=404
            )
        return JSONResponse(content={
            "game_id": game_id,
            "clv_results": results,
            "count": len(results),
        })
    except Exception as e:
        logger.error("Error computing CLV for game %d: %s", game_id, str(e), exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
