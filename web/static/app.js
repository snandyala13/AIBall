/* =====================================================
   PropTerminal - Frontend JavaScript (v3 - Bettor-Focused)
   ===================================================== */

// Track running analysis count for summary
var totalPicksFound = 0;
var totalUnitsFound = 0;
var allTopPicks = [];  // Collects best bets across all analyzed games

// Parlay builder state
var parlayLegs = [];  // [{player, prop_type, side, line, dk_odds, edge, prob, predicted_value}]

// Bet tracker state (persisted to localStorage)
var placedBets = [];
try {
    placedBets = JSON.parse(localStorage.getItem('aiBall_placedBets') || '[]');
} catch(e) { placedBets = []; }


/* =====================================================
   NAV STATS (sticky bar counts)
   ===================================================== */

function updateNavStats() {
    var picksEl = document.getElementById('nav-picks-count');
    var unitsEl = document.getElementById('nav-units-count');
    var picksStat = document.getElementById('nav-picks-stat');
    var unitsStat = document.getElementById('nav-units-stat');

    if (totalPicksFound > 0) {
        if (picksEl) picksEl.textContent = totalPicksFound;
        if (unitsEl) unitsEl.textContent = totalUnitsFound.toFixed(1) + 'u';
        if (picksStat) picksStat.style.display = 'flex';
        if (unitsStat) unitsStat.style.display = 'flex';
    } else {
        if (picksStat) picksStat.style.display = 'none';
        if (unitsStat) unitsStat.style.display = 'none';
    }
}


/* =====================================================
   BETTOR REASONING GENERATOR
   Human-readable "why" for each pick
   ===================================================== */

function generateBettorReasoning(pred) {
    var parts = [];
    var line = pred.line || 0;
    var propLabel = formatPropType(pred.prop_type);
    var pOver = pred.p_over || pred.our_probability || 0.5;
    var pUnder = pred.p_under || (1 - pOver);
    var confPct = Math.max(pOver, pUnder) * 100;
    var edgePct = Math.abs(pred.edge_percent || pred.edge || 0);

    // Classifier-native reasoning: edge-first framing
    var edgeLabel = edgePct >= 10 ? 'Strong' : (edgePct >= 5 ? 'Solid' : 'Moderate');
    // Compute breakeven from DK odds (what the bettor actually sees)
    var sideOdds = pred.side === 'under' ? (pred.under_odds || pred.dk_odds || -110) : (pred.over_odds || pred.dk_odds || -110);
    var breakeven;
    if (sideOdds > 0) { breakeven = 100 / (sideOdds + 100); }
    else { breakeven = Math.abs(sideOdds) / (Math.abs(sideOdds) + 100); }

    if (pred.side === 'under') {
        parts.push(edgeLabel + ' under signal on ' + line.toFixed(1) + ' ' + propLabel.toLowerCase() + '.');
        parts.push('Model gives ' + confPct.toFixed(0) + '% under vs ' + (breakeven * 100).toFixed(0) + '% needed — ' + edgePct.toFixed(1) + '% edge.');
    } else {
        parts.push(edgeLabel + ' over signal on ' + line.toFixed(1) + ' ' + propLabel.toLowerCase() + '.');
        parts.push('Model gives ' + confPct.toFixed(0) + '% over vs ' + (breakeven * 100).toFixed(0) + '% needed — ' + edgePct.toFixed(1) + '% edge.');
    }

    // Best odds callout — use line_shopping if available (most accurate)
    var bestBook = pred.best_book;
    var bestOdds = pred.best_odds;
    var dkOdds = pred.dk_odds;
    var shop = pred.line_shopping || [];
    if (shop.length > 0 && shop[0].book !== 'draftkings') {
        bestBook = shop[0].book;
        bestOdds = shop[0].odds;
    }
    if (bestBook && bestBook !== 'draftkings' && bestOdds && bestOdds > (dkOdds || -110)) {
        parts.push('Best price at ' + formatBookName(bestBook) + ' (' + formatOdds(bestOdds) + ').');
    }

    // Consensus and soft book context
    if (pred.soft_book_has_best && pred.best_book) {
        parts.push('Best price at soft book ' + formatBookName(pred.best_book) + ' — historically +6-7% ROI.');
    }
    if (pred.is_star_player && pred.side === 'under') {
        parts.push('Star player UNDER — stars hit unders at +2.3% ROI historically.');
    }
    if ((pred.line_range || 0) >= 1.0) {
        parts.push('Books disagree by ' + pred.line_range.toFixed(1) + ' pts — likely stale line.');
    }
    if (pred.consensus_edge && Math.abs(pred.consensus_edge) > 1) {
        parts.push('Sharp consensus: ' + (pred.consensus_edge > 0 ? '+' : '') + pred.consensus_edge.toFixed(1) + '% edge.');
    }

    // Opposite side explanation
    var oppSide = pred.side === 'over' ? 'under' : 'over';
    var oppOdds = pred.side === 'over' ? pred.under_odds : pred.over_odds;
    var oppProb = pred.side === 'over' ? pUnder : pOver;
    if (oppOdds && oppProb > 0) {
        var oppImplied;
        if (oppOdds > 0) { oppImplied = 100 / (oppOdds + 100); }
        else { oppImplied = Math.abs(oppOdds) / (Math.abs(oppOdds) + 100); }
        if (oppProb < oppImplied) {
            parts.push(oppSide.charAt(0).toUpperCase() + oppSide.slice(1) +
                ' at ' + formatOdds(oppOdds) + ' needs ' + (oppImplied * 100).toFixed(0) +
                '% but model gives ' + (oppProb * 100).toFixed(0) + '%.');
        }
    }

    return parts.join(' ');
}


/* =====================================================
   CONFIDENCE METER HELPER
   ===================================================== */

function getConfidencePercent(pred) {
    // Show the model's actual probability of the recommended side hitting
    // This is honest — a 55% model should show 55%, not 90%
    var prob = pred.our_probability || pred.prob || 0.5;
    return Math.round(prob * 100);
}

function getConfidenceLevel(pct) {
    if (pct >= 75) return 'high';
    if (pct >= 50) return 'medium';
    return 'low';
}

function renderConfidenceMeter(pred) {
    var pct = getConfidencePercent(pred);
    var level = getConfidenceLevel(pct);
    return '<div class="hero-confidence-meter">' +
        '<div class="confidence-meter">' +
            '<div class="confidence-fill conf-' + level + '" style="width:' + pct + '%"></div>' +
        '</div>' +
        '<span class="confidence-label">' + pct + '% confidence</span>' +
    '</div>';
}


/* =====================================================
   LINE SHOPPING SUMMARY HELPERS
   ===================================================== */

function computeGameShoppingSummary(picks) {
    var picksWithBetterOdds = 0;
    var evImprovements = [];
    var arbCount = 0;
    var middleCount = 0;

    picks.forEach(function(pred) {
        var bestBook = pred.best_book || 'draftkings';
        var bestOdds = pred.best_odds || pred.dk_odds || -110;
        var dkOdds = pred.dk_odds || -110;

        if (bestBook !== 'draftkings' && bestOdds > dkOdds) {
            picksWithBetterOdds++;
        }

        var oddsImpr = pred.odds_improvement || 0;
        if (oddsImpr > 0) {
            evImprovements.push(oddsImpr);
        }

        if (pred.arbitrage && pred.arbitrage.length > 0) {
            arbCount += pred.arbitrage.length;
        }
        if (pred.middles && pred.middles.length > 0) {
            middleCount += pred.middles.length;
        }
    });

    var avgEvImprovement = evImprovements.length > 0
        ? evImprovements.reduce(function(a, b) { return a + b; }, 0) / evImprovements.length
        : 0;

    return {
        picksWithBetterOdds: picksWithBetterOdds,
        avgEvImprovement: avgEvImprovement,
        arbCount: arbCount,
        middleCount: middleCount,
    };
}


/* =====================================================
   ANALYZE GAME (Dashboard inline results)
   ===================================================== */

function analyzeGame(gameId) {
    var container = document.getElementById('results-' + gameId);
    var card = document.getElementById('game-card-' + gameId);
    var btn = card.querySelector('.btn-analyze');

    // Ensure card is expanded
    card.classList.remove('collapsed');

    // Show skeleton loading state
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    container.innerHTML =
        '<div class="skeleton-loader">' +
            '<div class="skeleton-line"></div>' +
            '<div class="skeleton-line"></div>' +
            '<div class="skeleton-line"></div>' +
            '<div class="skeleton-line"></div>' +
        '</div>';

    fetch('/api/game/' + gameId + '/analyze')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            btn.disabled = false;
            btn.textContent = 'Re-Analyze';

            if (data.error) {
                container.innerHTML = '<div class="results-error">' + data.error + '</div>';
                return;
            }

            renderInlineResults(container, data);
            updateSummaryStats(data);
            collectTopPicks(data, gameId);

            // Update picks available badge
            var picksCount = 0;
            var v2Preds = (data.v2_predictions || []).filter(function(p) {
                return p.edge_percent > 0 && p.side !== 'pass';
            });
            picksCount = v2Preds.length || ((data.best_singles || []).length + (data.popoff_singles || []).length);
            var picksBadge = card.querySelector('.picks-available-badge');
            if (picksBadge && picksCount > 0) {
                picksBadge.textContent = picksCount + ' picks';
                picksBadge.style.display = 'inline-flex';
            }

            // Auto-fetch results for finished games
            if (card && card.dataset.status === 'final') {
                fetchInlineResults(gameId, container);
            }
        })
        .catch(function(err) {
            btn.disabled = false;
            btn.textContent = 'Retry Analysis';
            container.innerHTML = '<div class="results-error">Error: ' + err.message + '</div>';
        });
}

function fetchInlineResults(gameId, container) {
    fetch('/api/game/' + gameId + '/results')
        .then(function(r) {
            if (r.status === 404) return null;
            return r.json();
        })
        .then(function(data) {
            if (!data || data.error) return;
            if (typeof renderGradedResults === 'function') {
                renderGradedResults(container, data);
            }
        })
        .catch(function() {});
}


/* =====================================================
   RENDER INLINE RESULTS (Dashboard cards)
   ===================================================== */

function renderInlineResults(container, data) {
    var html = '';

    // v2 Model Predictions (shown first when available)
    var v2Preds = data.v2_predictions || [];
    var modelVersion = data.model_version || 'v1';
    if (v2Preds.length > 0) {
        // Filter to only actionable picks (edge > 0)
        var v2Picks = v2Preds.filter(function(p) {
            return p.edge_percent > 0 && p.side !== 'pass';
        });
        // Sort by edge descending
        v2Picks.sort(function(a, b) { return b.edge_percent - a.edge_percent; });
        // Show top 10
        v2Picks = v2Picks.slice(0, 10);

        if (v2Picks.length > 0) {
            // Compute shopping summary for this game's picks
            var shoppingStats = computeGameShoppingSummary(v2Picks);
            if (shoppingStats.picksWithBetterOdds > 0) {
                html += '<div class="shopping-summary-bar">' +
                    '<span class="shopping-icon">&#128722;</span> ' +
                    '<strong>Line Shopping:</strong> ' +
                    shoppingStats.picksWithBetterOdds + '/' + v2Picks.length + ' picks have better odds elsewhere' +
                    (shoppingStats.avgEvImprovement > 0 ? ' <span class="shopping-ev-gain">(avg +' + shoppingStats.avgEvImprovement.toFixed(1) + '% EV)</span>' : '') +
                    (shoppingStats.arbCount > 0 ? ' <span class="arb-alert">&#9888; ' + shoppingStats.arbCount + ' arb' + (shoppingStats.arbCount > 1 ? 's' : '') + '</span>' : '') +
                    (shoppingStats.middleCount > 0 ? ' <span class="middle-alert">' + shoppingStats.middleCount + ' middle' + (shoppingStats.middleCount > 1 ? 's' : '') + '</span>' : '') +
                '</div>';
            }

            html += '<div class="result-section v2-section">';
            html += '<div class="result-section-title">';
            html += 'Picks (' + v2Picks.length + ')';
            html += '</div>';
            v2Picks.forEach(function(pred) {
                html += renderV2CompactBet(pred);
            });
            html += '</div>';
        }
    }

    // v2 Parlays (shown after v2 picks when available)
    var v2Parlays = data.v2_parlays || [];
    if (v2Parlays.length > 0) {
        html += '<div class="result-section v2-section">';
        html += '<div class="result-section-title">';
        html += 'Suggested Parlays (' + v2Parlays.length + ')';
        html += '</div>';
        v2Parlays.forEach(function(parlay) {
            html += renderV2CompactParlay(parlay);
        });
        html += '</div>';
    }

    // Only show v1 stuff if v2 is not available
    if (v2Preds.length === 0) {
        var singles = data.best_singles || [];
        if (singles.length > 0) {
            html += '<div class="result-section">';
            html += '<div class="result-section-title">Best Singles (' + singles.length + ')</div>';
            singles.forEach(function(bet) {
                html += renderCompactBet(bet);
            });
            html += '</div>';
        }

        var popoffs = data.popoff_singles || [];
        if (popoffs.length > 0) {
            html += '<div class="result-section">';
            html += '<div class="result-section-title">Pop-Off Specials (' + popoffs.length + ')</div>';
            popoffs.forEach(function(bet) {
                html += renderCompactBet(bet);
            });
            html += '</div>';
        }

        var allParlays = (data.high_confidence_parlays || [])
            .concat(data.safe_parlays || [])
            .concat(data.standard_parlays || [])
            .concat(data.longshot_parlays || []);

        if (allParlays.length > 0) {
            html += '<div class="result-section">';
            html += '<div class="result-section-title">Parlays (' + allParlays.length + ')</div>';
            allParlays.forEach(function(parlay) {
                html += renderCompactParlay(parlay);
            });
            html += '</div>';
        }
    }

    // Line signals summary (before bets)
    var lineSignalsHtml = '';
    if (data.line_signals) {
        var sharpCount = 0;
        var crossBookCount = 0;
        for (var lsKey in data.line_signals) {
            var ls = data.line_signals[lsKey];
            if (ls.sharp_signal) sharpCount++;
            if (ls.cross_book_has_edge) crossBookCount++;
        }
        if (sharpCount > 0 || crossBookCount > 0) {
            lineSignalsHtml = '<div class="line-signals-inline">';
            if (sharpCount > 0) lineSignalsHtml += '<span class="line-signal-badge line-signal-sharp-agree">' + sharpCount + ' sharp</span> ';
            if (crossBookCount > 0) lineSignalsHtml += '<span class="line-signal-badge line-signal-cross-book">' + crossBookCount + ' cross-book</span>';
            lineSignalsHtml += '</div>';
        }
    }

    if (!html && !lineSignalsHtml) {
        html = '<div class="results-error">No value plays found for this game.</div>';
    }

    container.innerHTML = lineSignalsHtml + html;
}


/* =====================================================
   RENDER V2 ML PREDICTION (compact inline with bettor focus)
   ===================================================== */

function renderV2CompactBet(pred) {
    var edge = pred.edge_percent || 0;
    var edgeClass = Math.abs(edge) >= 8 ? 'edge-high' :
                    (Math.abs(edge) >= 5 ? 'edge-medium' : 'edge-low');
    var sideClass = pred.side === 'over' ? 'side-over' : 'side-under';
    var copyText = pred.player + ' ' + pred.side.toUpperCase() + ' ' + pred.line + ' ' + formatPropType(pred.prop_type);

    // Bettor reasoning instead of ML jargon
    var reasoning = generateBettorReasoning(pred);

    // Confidence meter
    var confPct = getConfidencePercent(pred);
    var confLevel = getConfidenceLevel(confPct);

    // Empirical filter tier badge
    var filterTier = pred.filter_tier || 'marginal';
    var filterBadgeHtml = '';
    var filterOpacity = '';
    if (filterTier === 'strong') {
        filterBadgeHtml = ' <span class="filter-badge filter-strong">HIGH HIT RATE</span>';
    } else if (filterTier === 'hidden') {
        filterBadgeHtml = ' <span class="filter-badge filter-hidden">LOW HIT RATE</span>';
        filterOpacity = ' style="opacity:0.5"';
    }

    // Strategy P badge (backtest-validated filter)
    var strategyBadgeHtml = '';
    if (pred.strategy_p) {
        strategyBadgeHtml = ' <span class="filter-badge filter-strong" title="Strategy P: edge 5-15%, confidence ≥55% — backtest-validated +5.4% ROI">STRATEGY P</span>';
    }

    // Cross-book signal badge
    var signalHtml = '';
    var crossSignal = pred.cross_book_signal || 'NEUTRAL';
    if (crossSignal === 'STRONG_UNDER' || crossSignal === 'STRONG_OVER') {
        signalHtml = ' <span class="line-signal-badge line-signal-cross-book">CROSS-BOOK</span>';
    }

    // Stale line badge
    var staleBadgeHtml = '';
    if (pred.is_stale && pred.staleness_score >= 60) {
        var staleClass = pred.staleness_score >= 80 ? 'stale-badge-critical' : 'stale-badge-warning';
        staleBadgeHtml = ' <span class="stale-line-badge ' + staleClass + '" title="' + escapeHtml(pred.stale_reason || 'Stale line detected') + '">STALE LINE</span>';
    }

    // Cascade badge (injury cascade boost)
    var cascadeBadgeHtml = '';
    if (pred.cascade) {
        var cBoost = pred.cascade.boost_pct || 0;
        var cSign = cBoost > 0 ? '+' : '';
        var cAbsent = pred.cascade.absent_player || '';
        cascadeBadgeHtml = ' <span class="cascade-badge-inline" title="' + escapeHtml(cAbsent) + ' out: ' + cSign + cBoost.toFixed(1) + '% historical boost">' +
            'CASCADE ' + cSign + cBoost.toFixed(0) + '%</span>';
    }

    // Arbitrage badge
    var arbHtml = '';
    if (pred.arbitrage && pred.arbitrage.length > 0) {
        var arb = pred.arbitrage[0];
        arbHtml = ' <span class="arb-badge">ARB ' + arb.arb_pct.toFixed(1) + '% (' +
            formatBookName(arb.book_a) + ' ' + arb.book_a_side + ' + ' +
            formatBookName(arb.book_b) + ' ' + arb.book_b_side + ')</span>';
    }

    // Middle opportunity badge
    var middleHtml = '';
    if (pred.middles && pred.middles.length > 0) {
        var mid = pred.middles[0];
        var midLabel = mid.compound_with_model ? 'COMPOUND MIDDLE' : 'MIDDLE';
        middleHtml = ' <span class="middle-badge' + (mid.compound_with_model ? ' middle-compound' : '') + '">' +
            midLabel + ' (' + formatBookName(mid.book) + ' ' + mid.line.toFixed(1) + ' vs ' + mid.consensus_line.toFixed(1) + ')</span>';
    }

    // Consensus-driven badges
    var softBookBadge = pred.soft_book_has_best ? ' <span class="soft-book-badge">SOFT BOOK</span>' : '';
    var starUnderBadge = (pred.is_star_player && pred.side === 'under') ? ' <span class="star-under-badge">STAR UNDER</span>' : '';
    var wideSpreadBadge = ((pred.line_range || 0) >= 1.0) ? ' <span class="wide-spread-badge">WIDE SPREAD</span>' : '';
    var shopItBadge = (!pred.soft_book_has_best && pred.best_book === 'draftkings' && (pred.line_shopping || []).length > 1)
        ? ' <span class="shop-it-badge">SHOP IT</span>' : '';

    // Cross-book best odds display
    var bestBookHtml = '';
    var allBooks = pred.all_books || {};
    var bestBook = pred.best_book || null;
    var bestOdds = pred.best_odds || null;

    if (allBooks && Object.keys(allBooks).length > 0) {
        var dkOdds = pred.dk_odds || -110;
        var foundBest = null;
        var foundBestOdds = null;
        var sideOddsKey = (pred.side === 'under') ? 'under_odds' : 'over_odds';

        for (var bk in allBooks) {
            if (bk === 'draftkings') continue;
            var bkData = allBooks[bk];
            var bkOdds = bkData[sideOddsKey] || bkData.odds || null;
            if (bkOdds != null && (foundBestOdds == null || bkOdds > foundBestOdds)) {
                foundBestOdds = bkOdds;
                foundBest = bk;
            }
        }

        if (!foundBest && bestBook && bestBook !== 'draftkings') {
            foundBest = bestBook;
            foundBestOdds = bestOdds;
        }

        if (foundBest && foundBestOdds != null && foundBestOdds > dkOdds) {
            var savedPct = pred.odds_improvement || 0;
            var tierClass = pred.odds_tier === 'premium' ? 'odds-premium' :
                           (pred.odds_tier === 'plus_money' ? 'odds-plus' :
                           (pred.odds_tier === 'reduced_vig' ? 'odds-reduced' : ''));
            bestBookHtml = ' <span class="better-line-badge ' + tierClass + '">' +
                formatBookName(foundBest) + ' ' + formatOdds(foundBestOdds) +
                (savedPct > 0.5 ? ' (+' + savedPct.toFixed(1) + '% edge)' : '') +
                '</span>';
        }
    }

    // DK odds display + edge comparison if better odds exist
    var edgeAtDk = pred.edge_at_dk || pred.edge_percent || 0;
    var edgeAtBest = pred.edge_percent || 0;
    var dkOddsHtml = '<span class="compact-bet-odds"><span class="mono dk-odds-value">' + formatOdds(pred.dk_odds) + '</span> DK</span>';

    // Line shopping expandable card (shows all books ranked by EV)
    var lineShoppingHtml = '';
    var lineShopping = pred.line_shopping || [];
    if (lineShopping.length > 1 || (allBooks && Object.keys(allBooks).length > 1)) {
        var shopItems = '';
        var shopData = lineShopping.length > 0 ? lineShopping : [];

        // If no pre-computed line_shopping, build from all_books
        if (shopData.length === 0 && allBooks) {
            var sideKey = (pred.side === 'under') ? 'under_odds' : 'over_odds';
            for (var bk in allBooks) {
                var bd = allBooks[bk];
                var bkOddsVal = bd[sideKey] || bd.odds || null;
                if (bkOddsVal != null) {
                    shopData.push({book: bk, line: bd.line || pred.line, odds: bkOddsVal, ev: 0});
                }
            }
            shopData.sort(function(a, b) { return (b.odds || 0) - (a.odds || 0); });
        }

        shopData.forEach(function(item, idx) {
            var isBest = (idx === 0 && item.odds > (pred.dk_odds || -110));
            var evStr = item.ev ? (item.ev > 0 ? '+' : '') + (item.ev * 100).toFixed(1) + '%' : '';
            shopItems += '<div class="shop-row' + (isBest ? ' shop-best' : '') + '">' +
                '<span class="shop-book">' + formatBookName(item.book) + '</span>' +
                '<span class="shop-line mono">' + (item.line || pred.line).toFixed(1) + '</span>' +
                '<span class="shop-odds mono">' + formatOdds(item.odds) + '</span>' +
                (evStr ? '<span class="shop-ev">' + evStr + ' EV</span>' : '') +
                '</div>';
        });

        if (shopItems) {
            lineShoppingHtml = '<div class="line-shopping-toggle" onclick="this.nextElementSibling.classList.toggle(\'expanded\');this.classList.toggle(\'open\')">' +
                '<span class="shop-icon">&#128176;</span> ' + shopData.length + ' books <span class="shop-arrow">&#9660;</span>' +
            '</div>' +
            '<div class="line-shopping-card">' + shopItems + '</div>';
        }
    }

    // Parlay builder button
    var parlayKey = pred.player + '|' + pred.prop_type + '|' + pred.side;
    var inParlay = parlayLegs.some(function(l) { return (l.player + '|' + l.prop_type + '|' + l.side) === parlayKey; });
    var parlayBtnClass = inParlay ? 'btn-add-parlay in-parlay' : 'btn-add-parlay';
    var parlayBtnText = inParlay ? 'In Parlay' : '+ Parlay';
    var parlayBtnHtml = '<button class="' + parlayBtnClass + '" onclick="toggleParlayLeg(this, \'' +
        escapeHtml(pred.player) + '\', \'' + pred.prop_type + '\', \'' + pred.side + '\', ' +
        pred.line + ', ' + (pred.dk_odds || -110) + ', ' + edge.toFixed(2) + ', ' +
        (pred.our_probability || 0.5).toFixed(4) + ', ' + (pred.predicted_value || 0).toFixed(2) +
        ')">' + parlayBtnText + '</button>';

    // Placed status
    var placedKey = pred.player + '|' + pred.prop_type + '|' + pred.side + '|' + pred.line;
    var isPlaced = placedBets.some(function(b) { return b.key === placedKey; });
    var placeBtnClass = isPlaced ? 'btn-place placed' : 'btn-place';
    var placeBtnText = isPlaced ? 'Placed' : 'Mark Placed';

    // Store analyzed line and player/prop for stale detection via data attributes
    return '<div class="compact-bet v2-bet" data-analyzed-line="' + pred.line + '" data-player="' + escapeHtml(pred.player) + '" data-prop-type="' + pred.prop_type + '" data-side="' + pred.side + '" data-filter-tier="' + filterTier + '"' + filterOpacity + '>' +
        '<div class="compact-bet-info">' +
            '<span class="compact-bet-player">' +
                '<a href="/player/' + encodeURIComponent(pred.player) + '">' + pred.player + '</a>' +
                strategyBadgeHtml + signalHtml + arbHtml + middleHtml + softBookBadge + starUnderBadge + wideSpreadBadge + shopItBadge + filterBadgeHtml + staleBadgeHtml + cascadeBadgeHtml +
            '</span>' +
            '<span class="compact-bet-prop">' +
                '<span class="side-badge ' + sideClass + '">' + pred.side.toUpperCase() + '</span> ' +
                '<span class="mono">' + pred.line.toFixed(1) + '</span> ' +
                formatPropType(pred.prop_type) +
            '</span>' +
            '<span class="v2-prediction" style="font-size:0.8rem;color:var(--text-secondary);line-height:1.5">' +
                reasoning +
            '</span>' +
            '<div style="margin:0.3rem 0">' +
                '<div class="confidence-meter">' +
                    '<div class="confidence-fill conf-' + confLevel + '" style="width:' + confPct + '%"></div>' +
                '</div>' +
                '<span class="confidence-label">' + confPct + '% confidence</span>' +
            '</div>' +
            dkOddsHtml + bestBookHtml +
            lineShoppingHtml +
            '<span class="compact-bet-units">' +
                '<span class="mono">' + (pred.recommended_units || 0).toFixed(1) + 'u</span>' +
                ' <button class="btn-copy" onclick="copyToClipboard(\'' + escapeHtml(copyText) + '\', event)" title="Copy bet text">Copy</button>' +
                ' ' + parlayBtnHtml +
                ' <button class="' + placeBtnClass + '" onclick="markAsPlaced(this, \'' + escapeHtml(placedKey) + '\', \'' +
                    escapeHtml(pred.player) + '\', \'' + pred.prop_type + '\', \'' + pred.side + '\', ' +
                    pred.line + ', ' + (pred.dk_odds || -110) + ', ' + (pred.recommended_units || 0).toFixed(2) +
                ')">' + placeBtnText + '</button>' +
            '</span>' +
        '</div>' +
        '<span class="compact-bet-edge ' + edgeClass + '"><span class="mono">' + edge.toFixed(1) + '%</span></span>' +
    '</div>';
}

function renderV2CompactParlay(parlay) {
    var legs = parlay.legs || [];
    var combinedOdds = parlay.combined_american_odds || 0;
    var combinedEdge = parlay.combined_edge || 0;
    var ev = parlay.ev_per_unit || 0;

    var html = '<div class="compact-parlay">';
    html += '<div class="compact-parlay-header">';
    html += '<span>' + legs.length + '-Leg Parlay</span>';
    html += '<span class="compact-parlay-odds">' +
        '<span class="mono">' + formatOdds(combinedOdds) + '</span>' +
        ' | Edge: <span class="mono">' + combinedEdge.toFixed(1) + '%</span>' +
        ' | EV: <span class="mono">' + (ev >= 0 ? '+' : '') + ev.toFixed(2) + 'u</span>' +
    '</span>';
    html += '</div>';

    var copyParts = [];
    legs.forEach(function(leg) {
        var sideClass = leg.side === 'over' ? 'side-over' : 'side-under';
        var legEdge = leg.edge_percent || 0;
        var legEdgeClass = legEdge >= 8 ? 'edge-high' : (legEdge >= 5 ? 'edge-medium' : 'edge-low');

        // Cross-book best odds for each leg
        var legBestHtml = '';
        var legBooks = leg.all_books || {};
        if (legBooks && Object.keys(legBooks).length > 0) {
            var bestBk = null;
            var bestBkOdds = null;
            var dkOdds = leg.dk_odds || -110;
            var legOddsKey = (leg.side === 'under') ? 'under_odds' : 'over_odds';
            for (var bk in legBooks) {
                if (bk === 'draftkings') continue;
                var bkOdds = legBooks[bk][legOddsKey] || legBooks[bk].odds || null;
                if (bkOdds != null && (bestBkOdds == null || bkOdds > bestBkOdds)) {
                    bestBkOdds = bkOdds;
                    bestBk = bk;
                }
            }
            if (bestBk && bestBkOdds != null && bestBkOdds > dkOdds) {
                legBestHtml = ' <span class="book-separator">|</span> <span class="book-odds book-best">' +
                    formatOdds(bestBkOdds) + ' ' + formatBookName(bestBk) + '</span>';
            }
        }

        html += '<div class="compact-parlay-leg">' +
            '<a href="/player/' + encodeURIComponent(leg.player) + '">' + leg.player + '</a>' +
            ' - ' + formatPropType(leg.prop_type) + ' ' +
            '<span class="side-badge ' + sideClass + '">' + leg.side.toUpperCase() + '</span> ' +
            '<span class="mono">' + leg.line + '</span>' +
            ' <span class="compact-leg-odds">' +
                '<span class="book-odds mono">' + formatOdds(leg.dk_odds) + ' DK</span>' +
                legBestHtml +
            '</span>' +
            ' <span class="compact-bet-edge ' + legEdgeClass + ' mono">' + legEdge.toFixed(1) + '%</span>' +
        '</div>';

        copyParts.push(leg.player + ' ' + leg.side.toUpperCase() + ' ' + leg.line + ' ' + formatPropType(leg.prop_type));
    });

    var copyText = copyParts.join(' + ');
    html += '<div class="compact-parlay-actions">' +
        '<button class="btn-copy" onclick="copyToClipboard(\'' + escapeHtml(copyText) + '\', event)">Copy Parlay</button>' +
    '</div>';
    html += '</div>';
    return html;
}

function renderCompactBet(bet) {
    var edgeClass = Math.abs(bet.edge_percent) >= 8 ? 'edge-high' :
                    (Math.abs(bet.edge_percent) >= 5 ? 'edge-medium' : 'edge-low');
    var sideClass = bet.side === 'over' ? 'side-over' : 'side-under';
    var copyText = bet.player + ' ' + bet.side.toUpperCase() + ' ' + bet.line + ' ' + formatPropType(bet.prop_type);
    var lineSignalHtml = renderLineSignalBadge(bet.line_signal);

    // Build cross-book odds display
    var oddsHtml = '<span class="mono">' + formatOdds(bet.dk_odds) + '</span> DK';
    var bestLineHtml = '';
    var bli = bet.best_line_info;
    if (bli && bli.all_books) {
        var bookParts = [];
        for (var bk in bli.all_books) {
            var bd = bli.all_books[bk];
            var isBest = (bk === bli.best_book);
            var bookLabel = formatBookName(bk);
            if (isBest && bk !== 'draftkings') {
                bookParts.push('<span class="book-odds book-best mono">' + formatOdds(bd.odds) + ' ' + bookLabel + '</span>');
            } else if (bk !== 'draftkings') {
                bookParts.push('<span class="book-odds mono">' + formatOdds(bd.odds) + ' ' + bookLabel + '</span>');
            }
        }
        if (bookParts.length > 0) {
            var dkPart = '<span class="book-odds' + (bli.best_book === 'draftkings' ? ' book-best' : '') + ' mono">' + formatOdds(bet.dk_odds) + ' DK</span>';
            oddsHtml = dkPart + ' <span class="book-separator">|</span> ' + bookParts.slice(0, 3).join(' <span class="book-separator">|</span> ');
        }
        if (bli.edge_vs_dk >= 1 && bli.best_book !== 'draftkings') {
            bestLineHtml = '<span class="better-line-badge">BETTER LINE: ' + formatBookName(bli.best_book) + ' ' + formatOdds(bli.best_odds) + '</span>';
        }
    }

    return '<div class="compact-bet">' +
        '<div class="compact-bet-info">' +
            '<span class="compact-bet-player">' +
                '<a href="/player/' + encodeURIComponent(bet.player) + '">' + bet.player + '</a>' +
                lineSignalHtml +
            '</span>' +
            '<span class="compact-bet-prop">' +
                '<span class="side-badge ' + sideClass + '">' + bet.side.toUpperCase() + '</span> ' +
                '<span class="mono">' + bet.line + '</span> ' +
                formatPropType(bet.prop_type) +
            '</span>' +
            '<span class="compact-bet-odds">' + oddsHtml + '</span>' +
            bestLineHtml +
            '<span class="compact-bet-units"><span class="mono">' + bet.recommended_units.toFixed(1) + 'u</span>' +
                ' <button class="btn-copy" onclick="copyToClipboard(\'' + escapeHtml(copyText) + '\', event)" title="Copy bet text">Copy</button>' +
            '</span>' +
        '</div>' +
        '<span class="compact-bet-edge ' + edgeClass + ' mono">' + bet.edge_percent.toFixed(1) + '%</span>' +
    '</div>';
}

function renderCompactParlay(parlay) {
    var html = '<div class="compact-parlay">';
    html += '<div class="compact-parlay-header">';
    html += '<span>' + parlay.legs.length + '-Leg Parlay</span>';
    html += '<span class="compact-parlay-odds mono">' + formatOdds(parlay.combined_dk_odds) + ' | Edge: ' + parlay.edge_percent.toFixed(1) + '% | ' + parlay.recommended_units.toFixed(1) + 'u</span>';
    html += '</div>';

    parlay.legs.forEach(function(leg) {
        var sideClass = leg.side === 'over' ? 'side-over' : 'side-under';
        var legSignalHtml = renderLineSignalBadge(leg.line_signal);
        var legOddsHtml = '<span class="book-odds mono">' + formatOdds(leg.dk_odds) + ' DK</span>';
        var bli = leg.best_line_info;
        if (bli && bli.best_book && bli.best_book !== 'draftkings') {
            legOddsHtml += ' <span class="book-separator">|</span> <span class="book-odds book-best mono">' +
                formatOdds(bli.best_odds) + ' ' + formatBookName(bli.best_book) + '</span>';
        }
        html += '<div class="compact-parlay-leg">' +
            '<a href="/player/' + encodeURIComponent(leg.player) + '">' + leg.player + '</a>' +
            legSignalHtml + ' - ' +
            formatPropType(leg.prop_type) + ' ' +
            '<span class="side-badge ' + sideClass + '">' + leg.side.toUpperCase() + '</span> ' +
            '<span class="mono">' + leg.line + '</span>' +
            ' <span class="compact-leg-odds">' + legOddsHtml + '</span>' +
        '</div>';
    });

    html += '</div>';
    return html;
}


/* =====================================================
   RENDER SINGLES (Game detail page)
   ===================================================== */

function renderSingles(container, singles) {
    container.innerHTML = '';
    singles.forEach(function(bet) {
        container.appendChild(createBetCard(bet, false));
    });
}


/* =====================================================
   RENDER POP-OFFS (Game detail page)
   ===================================================== */

function renderPopoffs(container, popoffs) {
    container.innerHTML = '';
    popoffs.forEach(function(bet) {
        container.appendChild(createBetCard(bet, true));
    });
}


/* =====================================================
   RENDER PARLAYS (Game detail page)
   ===================================================== */

function renderParlays(container, parlays) {
    container.innerHTML = '';
    parlays.forEach(function(parlay) {
        container.appendChild(createParlayCard(parlay));
    });
}


/* =====================================================
   BET CARD BUILDER (with edge gauge and copy button)
   ===================================================== */

function createBetCard(bet, isPopoff) {
    var card = document.createElement('div');
    var edgeAbs = Math.abs(bet.edge_percent);
    var edgeTier = edgeAbs >= 8 ? 'high' : (edgeAbs >= 5 ? 'medium' : 'low');
    var cardClass = 'bet-card';
    if (isPopoff) cardClass += ' popoff';
    else if (edgeTier === 'high') cardClass += ' edge-high-card';
    else if (edgeTier === 'medium') cardClass += ' edge-medium-card';
    card.className = cardClass;

    var edgeClass = 'edge-' + edgeTier;
    var sideClass = bet.side === 'over' ? 'side-over' : 'side-under';
    var copyText = bet.player + ' ' + bet.side.toUpperCase() + ' ' + bet.line + ' ' + formatPropType(bet.prop_type);
    var lineSignalHtml = renderLineSignalBadge(bet.line_signal);

    // Edge gauge bar (max at 15%)
    var gaugeWidth = Math.min(edgeAbs / 15 * 100, 100);
    var fillClass = edgeTier === 'high' ? 'fill-high' : (edgeTier === 'medium' ? 'fill-medium' : 'fill-low');

    // Build cross-book comparison table
    var bookComparisonHtml = '';
    var bli = bet.best_line_info;
    if (bli && bli.all_books && Object.keys(bli.all_books).length > 0) {
        var rows = '';
        var bookKeys = Object.keys(bli.all_books);
        bookKeys.sort(function(a, b) {
            if (a === 'draftkings') return -1;
            if (b === 'draftkings') return 1;
            if (a === bli.best_book) return -1;
            if (b === bli.best_book) return 1;
            return a.localeCompare(b);
        });
        bookKeys.forEach(function(bk) {
            var bd = bli.all_books[bk];
            var isBest = (bk === bli.best_book);
            var rowClass = isBest ? 'book-row book-row-best' : 'book-row';
            var starHtml = isBest ? ' <span class="book-star">&#9733;</span>' : '';
            var bestTag = (isBest && bk !== 'draftkings') ? ' <span class="book-best-tag">BEST</span>' : '';
            rows += '<div class="' + rowClass + '">' +
                '<span class="book-row-name">' + formatBookName(bk) + starHtml + bestTag + '</span>' +
                '<span class="book-row-odds mono">' + formatOdds(bd.odds) + '</span>' +
                '<span class="book-row-line mono">' + bd.line.toFixed(1) + '</span>' +
            '</div>';
        });

        var betterLineBadge = '';
        if (bli.edge_vs_dk >= 1 && bli.best_book !== 'draftkings') {
            betterLineBadge = '<div class="better-line-callout">' +
                'Best: ' + formatBookName(bli.best_book) + ' ' + formatOdds(bli.best_odds) +
                ' (' + bli.edge_vs_dk.toFixed(1) + 'pt better line)' +
            '</div>';
        }

        bookComparisonHtml =
            '<div class="book-comparison">' +
                '<div class="book-comparison-header">' +
                    '<span>Sportsbook</span><span>Odds</span><span>Line</span>' +
                '</div>' +
                rows +
                betterLineBadge +
            '</div>';
    }

    card.innerHTML =
        '<div class="bet-header">' +
            '<a href="/player/' + encodeURIComponent(bet.player) + '" class="bet-player">' + bet.player + '</a>' +
            lineSignalHtml +
            '<span class="bet-edge ' + edgeClass + ' mono">' + bet.edge_percent.toFixed(1) + '% edge</span>' +
        '</div>' +
        '<div class="bet-prop">' +
            '<span class="side-badge ' + sideClass + '">' + bet.side.toUpperCase() + '</span> ' +
            '<span class="mono">' + bet.line + '</span> ' +
            formatPropType(bet.prop_type) +
        '</div>' +
        '<div class="edge-gauge">' +
            '<div class="edge-gauge-bar">' +
                '<div class="edge-gauge-fill ' + fillClass + '" style="width:' + gaugeWidth + '%"></div>' +
            '</div>' +
            '<span class="edge-gauge-label mono">' + edgeAbs.toFixed(1) + '% edge</span>' +
        '</div>' +
        bookComparisonHtml +
        '<div class="bet-details">' +
            '<div class="bet-detail-item">' +
                '<span class="bet-detail-label">Projection</span>' +
                '<span class="bet-detail-value mono">' + bet.projected_value.toFixed(1) + '</span>' +
            '</div>' +
            '<div class="bet-detail-item">' +
                '<span class="bet-detail-label">DK Odds</span>' +
                '<span class="bet-detail-value bet-odds-value mono">' + formatOdds(bet.dk_odds) + '</span>' +
            '</div>' +
            '<div class="bet-detail-item">' +
                '<span class="bet-detail-label">Fair Odds</span>' +
                '<span class="bet-detail-value mono">' + formatOdds(bet.fair_odds) + '</span>' +
            '</div>' +
            '<div class="bet-detail-item">' +
                '<span class="bet-detail-label">Our Prob</span>' +
                '<span class="bet-detail-value mono">' + (bet.our_probability * 100).toFixed(1) + '%</span>' +
            '</div>' +
            '<div class="bet-detail-item">' +
                '<span class="bet-detail-label">Units</span>' +
                '<span class="bet-detail-value mono">' + bet.recommended_units.toFixed(1) + 'u</span>' +
            '</div>' +
        '</div>' +
        '<div class="bet-actions">' +
            '<div class="bet-reasoning" onclick="toggleReasoning(this)">' +
                '<span class="bet-reasoning-toggle">Show reasoning</span>' +
                '<div class="bet-reasoning-text">' + bet.reasoning + '</div>' +
            '</div>' +
            '<button class="btn-copy" onclick="copyToClipboard(\'' + escapeHtml(copyText) + '\', event)" title="Copy bet text">Copy</button>' +
        '</div>';

    return card;
}


/* =====================================================
   PARLAY CARD BUILDER
   ===================================================== */

function createParlayCard(parlay) {
    var card = document.createElement('div');
    card.className = 'parlay-card';

    var legsHtml = '';
    var copyParts = [];
    parlay.legs.forEach(function(leg) {
        var sideClass = leg.side === 'over' ? 'side-over' : 'side-under';
        var legSignalHtml = renderLineSignalBadge(leg.line_signal);

        var legBookHtml = '<span class="book-odds mono">' + formatOdds(leg.dk_odds) + ' DK</span>';
        var bli = leg.best_line_info;
        if (bli && bli.best_book && bli.best_book !== 'draftkings') {
            legBookHtml += ' <span class="book-separator">|</span> ' +
                '<span class="book-odds book-best mono">' + formatOdds(bli.best_odds) + ' ' + formatBookName(bli.best_book) + '</span>';
        }

        legsHtml +=
            '<div class="parlay-leg">' +
                '<span class="parlay-leg-player">' +
                    '<a href="/player/' + encodeURIComponent(leg.player) + '">' + leg.player + '</a>' +
                    legSignalHtml +
                '</span>' +
                '<span class="parlay-leg-prop">' + formatPropType(leg.prop_type) + '</span>' +
                '<span class="parlay-leg-side">' +
                    '<span class="side-badge ' + sideClass + '">' + leg.side.toUpperCase() + '</span> <span class="mono">' + leg.line + '</span>' +
                '</span>' +
                '<span class="parlay-leg-books">' + legBookHtml + '</span>' +
            '</div>';
        copyParts.push(leg.player + ' ' + leg.side.toUpperCase() + ' ' + leg.line + ' ' + formatPropType(leg.prop_type));
    });

    var copyText = copyParts.join(' + ');

    card.innerHTML =
        '<div class="parlay-header">' +
            '<span class="parlay-odds mono">' + formatOdds(parlay.combined_dk_odds) + '</span>' +
            '<div class="parlay-meta">' +
                '<span>Edge: <span class="mono">' + parlay.edge_percent.toFixed(1) + '%</span></span>' +
                '<span>Corr: <span class="mono">' + parlay.correlation.toFixed(2) + '</span></span>' +
                '<span class="mono">' + parlay.recommended_units.toFixed(1) + 'u</span>' +
                '<button class="btn-copy" onclick="copyToClipboard(\'' + escapeHtml(copyText) + '\', event)">Copy</button>' +
            '</div>' +
        '</div>' +
        '<div class="parlay-legs">' + legsHtml + '</div>' +
        (parlay.reasoning ?
            '<div class="parlay-reasoning">' + parlay.reasoning + '</div>' : '');

    return card;
}


/* =====================================================
   UTILITIES
   ===================================================== */

function formatOdds(odds) {
    if (!odds) return 'N/A';
    return odds > 0 ? '+' + odds : '' + odds;
}

function formatBookName(key) {
    var names = {
        'draftkings': 'DK',
        'fanduel': 'FD',
        'betmgm': 'BetMGM',
        'caesars': 'Caesars',
        'pointsbet': 'PointsBet',
        'bet365': 'Bet365',
        'unibet': 'Unibet',
        'williamhill': 'WH',
        'betrivers': 'BetRivers',
        'wynnbet': 'WynnBet',
        'superbook': 'SuperBook',
        'foxbet': 'FOX Bet',
        'twinspires': 'TwinSpires',
        'betus': 'BetUS',
        'bovada': 'Bovada',
        'mybookieag': 'MyBookie',
        'betonlineag': 'BetOnline',
        'lowvig': 'LowVig',
        'pinnacle': 'Pinnacle',
        'betway': 'Betway',
        'hardrockbet': 'Hard Rock',
        'fliff': 'Fliff',
        'espnbet': 'ESPN Bet',
        'fanatics': 'Fanatics',
        'betparx': 'BetParx',
        'windcreek': 'Wind Creek',
        'tipico_us': 'Tipico',
    };
    return names[key] || key.charAt(0).toUpperCase() + key.slice(1);
}

function formatPropType(prop) {
    var map = {
        'points': 'Points',
        'rebounds': 'Rebounds',
        'assists': 'Assists',
        'threes': '3PM',
        'steals': 'Steals',
        'blocks': 'Blocks',
        'turnovers': 'Turnovers',
        'pts_rebs_asts': 'PRA',
        'pts_rebs': 'Pts+Reb',
        'pts_asts': 'Pts+Ast',
        'rebs_asts': 'Reb+Ast',
        'stls_blks': 'Stl+Blk',
        'double_double': 'Double-Double',
        'triple_double': 'Triple-Double'
    };
    return map[prop] || prop;
}

function escapeHtml(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function toggleReasoning(el) {
    el.classList.toggle('expanded');
    var toggle = el.querySelector('.bet-reasoning-toggle');
    if (el.classList.contains('expanded')) {
        toggle.textContent = 'Hide reasoning';
    } else {
        toggle.textContent = 'Show reasoning';
    }
}

function copyToClipboard(text, event) {
    if (event) {
        event.stopPropagation();
        event.preventDefault();
    }

    navigator.clipboard.writeText(text).then(function() {
        var tooltip = document.getElementById('copy-tooltip');
        if (tooltip) {
            tooltip.textContent = 'Copied: ' + text;
            tooltip.classList.add('visible');
            setTimeout(function() {
                tooltip.classList.remove('visible');
            }, 2000);
        }

        if (event && event.target) {
            var btn = event.target;
            var original = btn.textContent;
            btn.textContent = 'Copied';
            btn.classList.add('copied');
            setTimeout(function() {
                btn.textContent = original;
                btn.classList.remove('copied');
            }, 1500);
        }
    }).catch(function() {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    });
}

function updateSummaryStats(data) {
    var v2Picks = (data.v2_predictions || []).filter(function(p) {
        return p.edge_percent > 0 && p.side !== 'pass';
    });

    if (v2Picks.length > 0) {
        totalPicksFound += v2Picks.length;
        v2Picks.forEach(function(p) {
            totalUnitsFound += p.recommended_units || 0;
        });
    } else {
        var singles = (data.best_singles || []).length + (data.popoff_singles || []).length;
        totalPicksFound += singles;
        totalUnitsFound += data.total_recommended_units || 0;
    }

    var picksEl = document.getElementById('picks-count');
    var unitsEl = document.getElementById('total-units');

    if (picksEl) picksEl.textContent = totalPicksFound;
    if (unitsEl) unitsEl.textContent = totalUnitsFound.toFixed(1) + 'u';

    updateNavStats();
}

function refreshData() {
    fetch('/api/refresh', { method: 'POST' })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.status === 'ok') {
                totalPicksFound = 0;
                totalUnitsFound = 0;
                allTopPicks = [];
                window.location.reload();
            }
        })
        .catch(function(err) {
            alert('Error refreshing: ' + err.message);
        });
}


/* =====================================================
   TOP PICKS AGGREGATION -> HERO PICKS
   ===================================================== */

function collectTopPicks(data, gameId) {
    var matchup = data.matchup || 'Game ' + gameId;

    // Remove any previous picks from this game
    allTopPicks = allTopPicks.filter(function(p) { return p.gameId !== gameId; });

    // Use v2 predictions when available, otherwise fall back to v1 singles
    var v2Preds = (data.v2_predictions || []).filter(function(p) {
        return p.edge_percent > 0 && p.side !== 'pass';
    });

    if (v2Preds.length > 0) {
        v2Preds.forEach(function(pred) {
            allTopPicks.push({
                type: 'v2_ml',
                gameId: gameId,
                matchup: matchup,
                player: pred.player,
                prop_type: pred.prop_type,
                side: pred.side,
                line: pred.line,
                dk_odds: pred.dk_odds,
                edge: pred.edge_percent,
                prob: pred.our_probability,
                units: pred.recommended_units || 0,
                projection: pred.predicted_value,
                p10: pred.p10 || 0,
                p90: pred.p90 || 0,
                confidence_stars: pred.confidence_stars,
                all_books: pred.all_books || {},
                best_book: pred.best_book,
                best_odds: pred.best_odds,
                odds_tier: pred.odds_tier || 'standard',
                odds_improvement: pred.odds_improvement || 0,
                edge_at_dk: pred.edge_at_dk || pred.edge_percent || 0,
                line_shopping: pred.line_shopping || [],
                arbitrage: pred.arbitrage || [],
                middles: pred.middles || [],
                filter_tier: pred.filter_tier || 'marginal',
                filter_reason: pred.filter_reason || '',
                staleness_score: pred.staleness_score || 0,
                is_stale: pred.is_stale || false,
                stale_reason: pred.stale_reason || '',
            });
        });
        renderHeroPicks();
        return;
    }

    // Fallback: v1 singles only if no v2
    (data.best_singles || []).forEach(function(bet) {
        allTopPicks.push({
            type: 'single',
            gameId: gameId,
            matchup: matchup,
            player: bet.player,
            prop_type: bet.prop_type,
            side: bet.side,
            line: bet.line,
            dk_odds: bet.dk_odds,
            edge: bet.edge_percent,
            prob: bet.our_probability,
            units: bet.recommended_units,
            projection: bet.projected_value,
            p10: 0,
            p90: 0,
            best_line_info: bet.best_line_info || null,
            all_books: {},
        });
    });

    (data.popoff_singles || []).forEach(function(bet) {
        allTopPicks.push({
            type: 'popoff',
            gameId: gameId,
            matchup: matchup,
            player: bet.player,
            prop_type: bet.prop_type,
            side: bet.side,
            line: bet.line,
            dk_odds: bet.dk_odds,
            edge: bet.edge_percent,
            prob: bet.our_probability,
            units: bet.recommended_units,
            projection: bet.projected_value,
            p10: 0,
            p90: 0,
            best_line_info: bet.best_line_info || null,
            all_books: {},
        });
    });

    renderHeroPicks();
}

function renderHeroPicks() {
    var section = document.getElementById('hero-picks-section');
    var grid = document.getElementById('hero-picks-grid');

    if (!section || !grid) return;

    if (allTopPicks.length === 0) {
        section.style.display = 'none';
        return;
    }

    // Deduplicate by player+prop+side, keeping highest edge
    var seen = {};
    var unique = [];
    allTopPicks.forEach(function(pick) {
        var key = pick.player + '|' + pick.prop_type + '|' + pick.side;
        if (!seen[key] || pick.edge > seen[key].edge) {
            seen[key] = pick;
        }
    });
    for (var k in seen) { unique.push(seen[k]); }

    // Sort by edge × confidence — a high-confidence 3% edge beats a low-confidence 10% edge
    // Filter out: 0% confidence picks, resolved picks (already hit or dead),
    // and "hidden" filter tier picks (low empirical hit rate)
    unique = unique.filter(function(p) {
        if ((p.prob || 0) <= 0.01) return false;
        if (p.resolved) return false;
        // Hidden tier picks don't make the top 5 hero picks
        if ((p.filter_tier || 'marginal') === 'hidden') return false;
        // If live data shows the stat already passed the line, this pick is cooked
        var lg = p.live_game;
        if (lg && lg.current_stat != null && lg.current_stat > p.line) {
            if (p.side === 'under') return false; // under is dead
            // over already hit — also no point showing it
            return false;
        }
        return true;
    });

    // Strategy P first: split into validated picks and the rest
    var stratPicks = unique.filter(function(p) { return p.strategy_p; });
    var otherPicks = unique.filter(function(p) { return !p.strategy_p; });

    // Balanced ranking: interleave across prop types instead of pure edge sort
    var byProp = {};
    stratPicks.forEach(function(p) {
        var pt = p.prop_type || '';
        if (!byProp[pt]) byProp[pt] = [];
        byProp[pt].push(p);
    });
    // Sort within each prop by confidence descending
    Object.keys(byProp).forEach(function(pt) {
        byProp[pt].sort(function(a, b) { return (b.prob || 0.5) - (a.prob || 0.5); });
    });
    // Interleave: rotate through prop types
    var propOrder = ['threes', 'points', 'assists', 'steals', 'blocks', 'pts_rebs_asts'];
    var available = propOrder.filter(function(pt) { return byProp[pt] && byProp[pt].length > 0; });
    var balanced = [];
    var idx = {};
    available.forEach(function(pt) { idx[pt] = 0; });
    var keepGoing = true;
    while (keepGoing) {
        keepGoing = false;
        available.forEach(function(pt) {
            if (idx[pt] < byProp[pt].length) {
                balanced.push(byProp[pt][idx[pt]]);
                idx[pt]++;
                keepGoing = true;
            }
        });
    }
    // Add any props not in propOrder
    Object.keys(byProp).forEach(function(pt) {
        if (propOrder.indexOf(pt) === -1) {
            byProp[pt].forEach(function(p) { balanced.push(p); });
        }
    });
    // Append non-Strategy-P picks after the validated ones
    balanced = balanced.concat(otherPicks);
    var top = balanced.slice(0, 50);

    var stratCount = stratPicks.length;
    var subtitle = document.getElementById('hero-picks-subtitle');
    if (subtitle) {
        subtitle.textContent = (stratCount > 0 ? stratCount + ' Strategy P picks' : 'Top ' + top.length + ' picks') + ' \u2022 updates every 30s';
    }

    var html = '';
    top.forEach(function(pick, i) {
        var confPct = Math.round((pick.prob || 0.5) * 100);
        var confLevel = getConfidenceLevel(confPct);
        var pickEdge = Math.abs(pick.edge || pick.edge_percent || 0);
        var confLabelText = pickEdge >= 10 ? 'STRONG EDGE' : (pickEdge >= 5 ? 'SOLID EDGE' : 'MODERATE EDGE');
        var sideClass = pick.side === 'over' ? 'over' : 'under';
        var propLabel = formatPropType(pick.prop_type);

        // Generate bettor reasoning (use shared function)
        var reasoning = generateBettorReasoning(pick);

        // Build odds rows from all_books
        var oddsRowsHtml = '';
        var allBooks = pick.all_books || {};
        var pickSide = pick.side || 'over';
        var oddsKey = pickSide === 'under' ? 'under_odds' : 'over_odds';
        var bookKeys = Object.keys(allBooks);
        if (bookKeys.length > 0) {
            // Sort by best odds for the recommended side
            bookKeys.sort(function(a, b) {
                var aOdds = allBooks[a][oddsKey] || allBooks[a].odds || -999;
                var bOdds = allBooks[b][oddsKey] || allBooks[b].odds || -999;
                return bOdds - aOdds;
            });
            bookKeys.slice(0, 3).forEach(function(bk, j) {
                var bkData = allBooks[bk];
                var bkOdds = bkData[oddsKey] || bkData.odds || null;
                var isBest = (j === 0 && bkOdds != null && bookKeys.length > 1);
                var rowClass = isBest ? 'hero-odds-row best' : 'hero-odds-row';
                var oddsClass = (bk === 'draftkings') ? 'odds dk-odds-value' : 'odds';
                oddsRowsHtml += '<div class="' + rowClass + '">' +
                    '<span class="book-name">' + formatBookName(bk) + '</span>' +
                    '<span class="' + oddsClass + '">' + formatOdds(bkOdds) + '</span>' +
                    (isBest ? '<span class="badge">BEST</span>' : '') +
                '</div>';
            });
        }
        // Always show DK if not in all_books
        if (!allBooks['draftkings'] && pick.dk_odds) {
            oddsRowsHtml = '<div class="hero-odds-row">' +
                '<span class="book-name">DK</span>' +
                '<span class="odds dk-odds-value">' + formatOdds(pick.dk_odds) + '</span>' +
            '</div>' + oddsRowsHtml;
        }

        var copyText = pick.player + ' ' + pick.side.toUpperCase() + ' ' + pick.line + ' ' + propLabel;

        // Placed status
        var placedKey = pick.player + '|' + pick.prop_type + '|' + pick.side + '|' + pick.line;
        var isPlaced = placedBets.some(function(b) { return b.key === placedKey; });
        var placeBtnClass = isPlaced ? 'btn-place placed' : 'btn-place';
        var placeBtnText = isPlaced ? 'Placed' : 'Mark Placed';

        html += '<div class="hero-pick" data-analyzed-line="' + pick.line + '" data-player="' + escapeHtml(pick.player) + '" data-prop-type="' + pick.prop_type + '" data-side="' + pick.side + '" data-game-id="' + pick.gameId + '">' +
            '<div class="hero-pick-header">' +
                '<span class="hero-rank">#' + (i + 1) + '</span>' +
                '<span class="hero-confidence ' + confLevel + '">' + confLabelText + '</span>' +
            '</div>' +
            '<div class="hero-pick-body">' +
                '<div class="hero-player"><a href="/player/' + encodeURIComponent(pick.player) + '">' + pick.player + '</a>' +
                    (pick.arbitrage && pick.arbitrage.length > 0 ? ' <span class="arb-badge">ARB</span>' : '') +
                    (pick.middles && pick.middles.length > 0 && pick.middles[0].compound_with_model ? ' <span class="middle-badge middle-compound">COMPOUND MIDDLE</span>' : '') +
                    (pick.middles && pick.middles.length > 0 && !pick.middles[0].compound_with_model ? ' <span class="middle-badge">MIDDLE</span>' : '') +
                    (pick.soft_book_has_best ? ' <span class="soft-book-badge">SOFT BOOK</span>' : '') +
                    (pick.is_star_player && pick.side === 'under' ? ' <span class="star-under-badge">STAR UNDER</span>' : '') +
                    ((pick.line_range || 0) >= 1.0 ? ' <span class="wide-spread-badge">WIDE SPREAD</span>' : '') +
                    (!pick.soft_book_has_best && pick.best_book === 'draftkings' && (pick.line_shopping || []).length > 1 ? ' <span class="shop-it-badge">SHOP IT</span>' : '') +
                '</div>' +
                '<div class="hero-matchup-label">' + pick.matchup + '</div>' +
                '<div class="hero-prop">' +
                    '<span class="hero-side ' + sideClass + '">' + pick.side.toUpperCase() + '</span>' +
                    '<span class="hero-line">' + pick.line.toFixed(1) + ' ' + propLabel + '</span>' +
                '</div>' +
                (function() {
                    var lg = pick.live_game;
                    if (!lg) return '';
                    var gs = lg.game_status || {};
                    var clk = gs.time_remaining || gs.clock || gs.raw || '';
                    var per = gs.period_label || gs.raw || ('Q' + (gs.period || ''));
                    var stat = lg.current_stat || 0;
                    var proj = (lg.projected_final || 0).toFixed(1);
                    var onP = lg.on_pace;
                    var progPct = Math.min(lg.progress_pct || 0, 200);
                    var barCol = progPct >= 100 ? '#00e676' : (progPct >= 70 ? '#ff9800' : 'var(--accent)');
                    return '<div class="live-inline-tracker">' +
                        '<div class="live-inline-row">' +
                            '<span class="live-inline-badge">LIVE</span>' +
                            '<span class="live-inline-clock">' + per + ' ' + clk + '</span>' +
                            '<span class="live-inline-stat">Currently: <strong>' + stat + '</strong></span>' +
                            '<span class="live-inline-proj">Proj: ' + proj + '</span>' +
                            '<span class="live-inline-pace ' + (onP ? 'pace-on' : 'pace-off') + '">' + (onP ? 'ON PACE' : 'BEHIND') + '</span>' +
                        '</div>' +
                        '<div class="live-inline-bar"><div class="live-inline-fill" style="width:' + Math.min(progPct, 100) + '%;background:' + barCol + '"></div></div>' +
                    '</div>';
                })() +
                '<div class="hero-why">' + reasoning + '</div>' +
                renderConfidenceMeter({our_probability: pick.prob, edge_percent: pick.edge}) +
                (oddsRowsHtml ? '<div class="hero-odds">' + oddsRowsHtml + '</div>' : '') +
            '</div>' +
            '<div class="hero-pick-footer">' +
                '<span class="hero-edge">+' + pick.edge.toFixed(1) + '% edge' +
                    (pick.odds_improvement > 0.5 ? ' <span class="odds-boost">(+' + pick.odds_improvement.toFixed(1) + '% from shopping)</span>' : '') +
                '</span>' +
                '<span class="hero-units">' + pick.units.toFixed(1) + 'u</span>' +
                '<button class="btn-copy" onclick="copyToClipboard(\'' + escapeHtml(copyText) + '\', event)">Copy</button>' +
                '<button class="' + placeBtnClass + '" onclick="markAsPlaced(this, \'' + escapeHtml(placedKey) + '\', \'' +
                    escapeHtml(pick.player) + '\', \'' + pick.prop_type + '\', \'' + pick.side + '\', ' +
                    pick.line + ', ' + (pick.dk_odds || -110) + ', ' + (pick.units || 0).toFixed(2) +
                ')">' + placeBtnText + '</button>' +
            '</div>' +
        '</div>';
    });

    grid.innerHTML = html;
    section.style.display = 'block';

    // Also render the old top-picks section if it exists (for backward compat)
    var oldSection = document.getElementById('top-picks-section');
    if (oldSection) oldSection.style.display = 'none';

    // Smooth scroll to hero picks
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Keep renderTopPicks as alias for backward compat (called from dashboard.html collectTopPicks)
function renderTopPicks() {
    renderHeroPicks();
}


/* =====================================================
   BET TRACKER - Mark Placed / Track P&L
   ===================================================== */

function markAsPlaced(btn, key, player, propType, side, line, dkOdds, units) {
    var existing = placedBets.findIndex(function(b) { return b.key === key; });

    if (existing !== -1) {
        // Remove
        placedBets.splice(existing, 1);
        btn.classList.remove('placed');
        btn.textContent = 'Mark Placed';
    } else {
        // Add
        placedBets.push({
            key: key,
            player: player,
            prop_type: propType,
            side: side,
            line: line,
            dk_odds: dkOdds,
            units: units,
            timestamp: new Date().toISOString(),
            result: null, // null = pending, 'hit', 'miss', 'push'
        });
        btn.classList.add('placed');
        btn.textContent = 'Placed';
    }

    try {
        localStorage.setItem('aiBall_placedBets', JSON.stringify(placedBets));
    } catch(e) {}

    renderBetTracker();
}

function renderBetTracker() {
    var tracker = document.getElementById('bet-tracker');
    if (!tracker) return;

    if (placedBets.length === 0) {
        tracker.style.display = 'none';
        return;
    }

    tracker.style.display = 'flex';

    var totalBets = placedBets.length;
    var hits = placedBets.filter(function(b) { return b.result === 'hit'; }).length;
    var misses = placedBets.filter(function(b) { return b.result === 'miss'; }).length;
    var pending = placedBets.filter(function(b) { return !b.result; }).length;
    var totalUnits = 0;
    placedBets.forEach(function(b) { totalUnits += (b.units || 0); });

    // Simple P&L estimate (assumes -110 odds for simplification)
    var pnl = 0;
    placedBets.forEach(function(b) {
        if (b.result === 'hit') {
            var odds = b.dk_odds || -110;
            if (odds > 0) {
                pnl += (b.units || 1) * (odds / 100);
            } else {
                pnl += (b.units || 1) * (100 / Math.abs(odds));
            }
        } else if (b.result === 'miss') {
            pnl -= (b.units || 1);
        }
    });

    var pnlClass = pnl > 0 ? 'positive' : (pnl < 0 ? 'negative' : 'neutral');
    var pnlStr = pnl >= 0 ? '+' + pnl.toFixed(1) + 'u' : pnl.toFixed(1) + 'u';

    var record = '';
    if (hits > 0 || misses > 0) {
        record = hits + '-' + misses;
    }

    var statsHtml = '<span class="tracker-stat neutral">' + totalBets + ' placed</span>';
    if (record) {
        statsHtml += '<span class="tracker-stat neutral">' + record + '</span>';
    }
    if (pending > 0) {
        statsHtml += '<span class="tracker-stat neutral">' + pending + ' pending</span>';
    }
    statsHtml += '<span class="tracker-stat ' + pnlClass + '">' + pnlStr + '</span>';
    statsHtml += '<span class="tracker-stat neutral">' + totalUnits.toFixed(1) + 'u wagered</span>';

    document.getElementById('bet-tracker-stats').innerHTML = statsHtml;
}

function clearBetTracker() {
    if (!confirm('Clear all placed bets for tonight?')) return;
    placedBets = [];
    try {
        localStorage.setItem('aiBall_placedBets', JSON.stringify(placedBets));
    } catch(e) {}
    renderBetTracker();
}


/* =====================================================
   PARLAY BUILDER
   ===================================================== */

function toggleParlayLeg(btn, player, propType, side, line, dkOdds, edge, prob, predicted) {
    var key = player + '|' + propType + '|' + side;
    var existingIdx = parlayLegs.findIndex(function(l) { return (l.player + '|' + l.prop_type + '|' + l.side) === key; });

    if (existingIdx !== -1) {
        parlayLegs.splice(existingIdx, 1);
        btn.classList.remove('in-parlay');
        btn.textContent = '+ Parlay';
    } else {
        parlayLegs.push({
            player: player,
            prop_type: propType,
            side: side,
            line: line,
            dk_odds: dkOdds,
            edge: edge,
            prob: prob,
            predicted_value: predicted,
        });
        btn.classList.add('in-parlay');
        btn.textContent = 'In Parlay';
    }

    renderParlaySlip();
}

function removeParlayLeg(index) {
    var removed = parlayLegs.splice(index, 1)[0];
    if (removed) {
        // Un-highlight the button in the bet list
        var key = removed.player + '|' + removed.prop_type + '|' + removed.side;
        var btns = document.querySelectorAll('.btn-add-parlay.in-parlay');
        btns.forEach(function(b) {
            // Check if this button matches by checking its onclick
            var onclick = b.getAttribute('onclick') || '';
            if (onclick.indexOf(removed.player) !== -1 && onclick.indexOf(removed.prop_type) !== -1) {
                b.classList.remove('in-parlay');
                b.textContent = '+ Parlay';
            }
        });
    }
    renderParlaySlip();
}

function renderParlaySlip() {
    var slip = document.getElementById('parlay-slip');
    if (!slip) return;

    if (parlayLegs.length === 0) {
        slip.style.display = 'none';
        return;
    }

    slip.style.display = 'block';

    var legsHtml = '';
    parlayLegs.forEach(function(leg, i) {
        var sideClass = leg.side === 'over' ? 'side-over' : 'side-under';
        legsHtml += '<div class="parlay-slip-leg">' +
            '<div class="parlay-slip-leg-info">' +
                '<span class="side-badge ' + sideClass + '">' + leg.side.toUpperCase() + '</span> ' +
                '<strong>' + leg.player + '</strong> ' +
                '<span class="mono">' + leg.line.toFixed(1) + '</span> ' +
                formatPropType(leg.prop_type) +
                ' <span class="mono" style="color:var(--text-muted)">' + formatOdds(leg.dk_odds) + '</span>' +
            '</div>' +
            '<button class="btn-remove" onclick="removeParlayLeg(' + i + ')" title="Remove">&times;</button>' +
        '</div>';
    });

    document.getElementById('parlay-slip-legs').innerHTML = legsHtml;

    // Calculate combined odds (multiply decimal odds)
    var combinedDecimal = 1;
    var combinedProb = 1;
    var combinedEdge = 0;

    parlayLegs.forEach(function(leg) {
        var decOdds;
        if (leg.dk_odds > 0) {
            decOdds = 1 + (leg.dk_odds / 100);
        } else {
            decOdds = 1 + (100 / Math.abs(leg.dk_odds));
        }
        combinedDecimal *= decOdds;
        combinedProb *= (leg.prob || 0.5);
    });

    // Convert back to American
    var combinedAmerican;
    if (combinedDecimal >= 2) {
        combinedAmerican = Math.round((combinedDecimal - 1) * 100);
    } else {
        combinedAmerican = Math.round(-100 / (combinedDecimal - 1));
    }

    var impliedProb = 1 / combinedDecimal;
    combinedEdge = ((combinedProb - impliedProb) / impliedProb * 100);
    var winProb = Math.round(combinedProb * 100);

    document.getElementById('parlay-combined-odds').textContent = formatOdds(combinedAmerican);
    document.getElementById('parlay-win-prob').textContent = winProb + '%';
    document.getElementById('parlay-edge').textContent = (combinedEdge >= 0 ? '+' : '') + combinedEdge.toFixed(1) + '%';

    // Build copy text
    var copyParts = parlayLegs.map(function(l) {
        return l.player + ' ' + l.side.toUpperCase() + ' ' + l.line.toFixed(1) + ' ' + formatPropType(l.prop_type);
    });
    var copyFull = copyParts.join(' + ') + ' | ' + formatOdds(combinedAmerican);
    document.getElementById('parlay-copy-text').setAttribute('data-copy', copyFull);
}

function copyParlaySlip() {
    var el = document.getElementById('parlay-copy-text');
    var text = el ? el.getAttribute('data-copy') : '';
    if (text) {
        copyToClipboard(text);
    }
}

function clearParlaySlip() {
    parlayLegs = [];
    // Un-highlight all parlay buttons
    var btns = document.querySelectorAll('.btn-add-parlay.in-parlay');
    btns.forEach(function(b) {
        b.classList.remove('in-parlay');
        b.textContent = '+ Parlay';
    });
    renderParlaySlip();
}


/* =====================================================
   LINE SIGNAL BADGES
   ===================================================== */

function renderLineSignalBadge(signal) {
    if (!signal || signal === 'neutral') return '';

    var badgeMap = {
        'sharp_agree':     { text: 'SHARP',   cls: 'line-signal-sharp-agree',   title: 'Sharp money agrees with model' },
        'combined_agree':  { text: 'SHARP+',  cls: 'line-signal-combined',      title: 'Sharp money + cross-book edge both agree' },
        'cross_book_edge': { text: 'EDGE',    cls: 'line-signal-cross-book',    title: 'Cross-book line discrepancy favors this bet' },
        'sharp_disagree':  { text: 'CAUTION', cls: 'line-signal-sharp-disagree', title: 'Sharp money disagrees with model' },
    };

    var info = badgeMap[signal];
    if (!info) return '';

    return ' <span class="line-signal-badge ' + info.cls + '" title="' + info.title + '">' + info.text + '</span>';
}


/* =====================================================
   CLV STATS RENDERING
   ===================================================== */

function renderClvStats(data) {
    var clv = data.clv_stats;
    if (!clv) return;

    var container = document.getElementById('clv-stats-container');
    if (!container) return;

    container.innerHTML =
        '<div class="clv-stats">' +
            '<div class="clv-stats-title">Closing Line Value</div>' +
            '<div class="clv-stats-row">' +
                '<div class="clv-stat">' +
                    '<span class="clv-stat-value ' + (clv.avg_clv >= 0 ? 'positive' : 'negative') + ' mono">' +
                        (clv.avg_clv >= 0 ? '+' : '') + clv.avg_clv.toFixed(2) + '</span>' +
                    '<span class="clv-stat-label">Avg CLV</span>' +
                '</div>' +
                '<div class="clv-stat">' +
                    '<span class="clv-stat-value mono">' + clv.clv_positive_pct.toFixed(1) + '%</span>' +
                    '<span class="clv-stat-label">Beat Close</span>' +
                '</div>' +
                '<div class="clv-stat">' +
                    '<span class="clv-stat-value mono">' + clv.predictions_with_clv + '</span>' +
                    '<span class="clv-stat-label">Tracked</span>' +
                '</div>' +
            '</div>' +
        '</div>';

    container.style.display = 'block';
}


/* =====================================================
   LIVE LINE POLLING
   ===================================================== */

var LIVE_POLL_INTERVAL = 30 * 1000;  // 30 seconds (real-time edge recalc is cheap)
var livePollTimer = null;
var livePollingActive = false;
var previousLines = {};
var lastScanTimestamp = null;
var scanAgoTimer = null;

function startLivePolling() {
    if (livePollingActive) return;
    livePollingActive = true;

    pollLiveLines();
    livePollTimer = setInterval(pollLiveLines, LIVE_POLL_INTERVAL);

    if (scanAgoTimer) clearInterval(scanAgoTimer);
    scanAgoTimer = setInterval(updateScanAgoText, 5000);

    console.log('Live line polling started (interval: ' + (LIVE_POLL_INTERVAL / 1000) + 's)');
}

function stopLivePolling() {
    livePollingActive = false;
    if (livePollTimer) {
        clearInterval(livePollTimer);
        livePollTimer = null;
    }
    if (scanAgoTimer) {
        clearInterval(scanAgoTimer);
        scanAgoTimer = null;
    }
    console.log('Live line polling stopped');
}

function setLivePollInterval(ms) {
    LIVE_POLL_INTERVAL = ms;
    if (livePollingActive) {
        stopLivePolling();
        startLivePolling();
    }
}

function pollLiveLines() {
    var dateParam = '';
    if (typeof currentDate !== 'undefined' && currentDate) {
        dateParam = '?date=' + currentDate;
    }

    // Try real-time predictions first (recalculated edge with fresh odds)
    fetch('/api/live-predictions' + dateParam)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.status === 'ok' && data.games) {
                lastScanTimestamp = data.timestamp;
                updatePicksWithFreshEdge(data.games);
                updateMovementTicker(data.movements || []);
                updateLastScanTime(data.timestamp);
                updateLiveStatusIndicator(data);
                var recalcMs = data.recalc_time_ms || 0;
                console.log('Live predictions updated: ' + data.total_predictions + ' props in ' + recalcMs + 'ms');
                return;
            }
            // Fall back to raw live lines if not analyzed yet
            return pollLiveLinesLegacy(dateParam);
        })
        .catch(function(err) {
            console.warn('Live predictions failed, falling back:', err.message);
            pollLiveLinesLegacy(dateParam);
        });
}

function pollLiveLinesLegacy(dateParam) {
    fetch('/api/live-lines' + dateParam)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) return;
            lastScanTimestamp = data.timestamp;
            updateLiveLines(data);
            updateMovementTicker(data.movements || []);
            updateCrossBookEdges(data.cross_book_edges || []);
            updateLastScanTime(data.timestamp);
        })
        .catch(function(err) {
            console.warn('Live line poll failed:', err.message);
        });
}

function updatePicksWithFreshEdge(gamesData) {
    // gamesData = {game_id: {predictions: [...], count: N}}
    // Update all displayed picks with fresh edge/odds from real-time recalculation

    // Build a lookup: player|prop_type -> prediction
    var freshLookup = {};
    for (var gameId in gamesData) {
        var preds = gamesData[gameId].predictions || [];
        preds.forEach(function(pred) {
            var key = pred.player + '|' + pred.prop_type;
            freshLookup[key] = pred;
        });
    }

    // Update all pick elements with data attributes
    var pickElements = document.querySelectorAll('[data-analyzed-line]');
    pickElements.forEach(function(el) {
        var player = el.dataset.player;
        var propType = el.dataset.propType;
        if (!player || !propType) return;

        var key = player + '|' + propType;
        var fresh = freshLookup[key];
        if (!fresh) return;

        // Clear stale badges — picks are now auto-healed
        clearStaleBadge(el);

        // Always sync displayed line with fresh data
        var propSpan = el.querySelector('.compact-bet-prop');
        if (propSpan) {
            var monoEl = propSpan.querySelector('.mono');
            if (monoEl) {
                var displayedLine = monoEl.textContent.trim();
                var freshLine = fresh.line.toFixed(1);
                if (displayedLine !== freshLine) {
                    monoEl.textContent = freshLine;
                    monoEl.classList.remove('line-changed');
                    void monoEl.offsetWidth;
                    monoEl.classList.add('line-changed');
                }
            }
            // Show delta from original analysis line
            var lineDelta = fresh.line_moved || 0;
            if (lineDelta !== 0 && monoEl) {
                var deltaEl = propSpan.querySelector('.line-delta');
                if (!deltaEl) {
                    deltaEl = document.createElement('span');
                    deltaEl.className = 'line-delta';
                    monoEl.parentNode.insertBefore(deltaEl, monoEl.nextSibling);
                }
                var arrow = lineDelta > 0 ? '\u2191' : '\u2193';
                deltaEl.className = 'line-delta ' + (lineDelta > 0 ? 'line-up' : 'line-down');
                deltaEl.textContent = ' (' + (lineDelta > 0 ? '+' : '') + lineDelta.toFixed(1) + arrow + ')';
            }
        }
        el.dataset.analyzedLine = String(fresh.line);

        // Update edge display with flash
        var edgeEl = el.querySelector('.compact-bet-edge .mono') || el.querySelector('.compact-bet-edge');
        if (edgeEl) {
            var oldEdge = parseFloat(edgeEl.textContent) || 0;
            var newEdge = fresh.edge_percent || 0;
            if (Math.abs(newEdge - oldEdge) > 0.05) {
                edgeEl.textContent = newEdge.toFixed(1) + '%';
                edgeEl.classList.remove('edge-flash-up', 'edge-flash-down');
                void edgeEl.offsetWidth;
                edgeEl.classList.add(newEdge > oldEdge ? 'edge-flash-up' : 'edge-flash-down');
            }
        }

        // Update odds display with direction-aware flash
        var side = fresh.side || el.dataset.side;
        updatePickOddsAnimated(el, fresh.over_odds, fresh.under_odds, side);

        // Update best book badge with animation
        var bestBookEl = el.querySelector('.best-book-badge, .better-line-badge');
        if (fresh.best_book && fresh.best_book !== 'draftkings') {
            var bestOdds = fresh.best_odds || '';
            var bestText = formatBookName(fresh.best_book) + ' ' + formatOdds(bestOdds);
            if (bestBookEl) {
                if (bestBookEl.textContent.trim() !== bestText.trim()) {
                    bestBookEl.textContent = bestText;
                    bestBookEl.classList.remove('book-changed');
                    void bestBookEl.offsetWidth;
                    bestBookEl.classList.add('book-changed');
                }
                bestBookEl.style.display = '';
            }
        }

        // Update line shopping data attribute for expandable card
        if (fresh.line_shopping && fresh.line_shopping.length > 0) {
            el.dataset.lineShopping = JSON.stringify(fresh.line_shopping);
        }

        // Show source indicator for real-time recalc
        var sourceEl = el.querySelector('.source-badge');
        if (sourceEl && fresh.source === 'v2_realtime_recalc') {
            sourceEl.textContent = 'LIVE';
            sourceEl.className = 'source-badge source-live';
        }

        // Update empirical filter tier from real-time data
        var freshTier = fresh.filter_tier || 'marginal';
        el.dataset.filterTier = freshTier;
        if (freshTier === 'hidden') {
            el.style.opacity = '0.5';
        } else {
            el.style.opacity = '';
        }
        // Update filter badge
        var existingBadge = el.querySelector('.filter-badge');
        if (freshTier === 'strong') {
            if (existingBadge) {
                existingBadge.className = 'filter-badge filter-strong';
                existingBadge.textContent = 'HIGH HIT RATE';
            } else {
                var playerEl = el.querySelector('.compact-bet-player');
                if (playerEl) {
                    var badge = document.createElement('span');
                    badge.className = 'filter-badge filter-strong';
                    badge.textContent = 'HIGH HIT RATE';
                    playerEl.appendChild(badge);
                }
            }
        } else if (freshTier === 'hidden') {
            if (existingBadge) {
                existingBadge.className = 'filter-badge filter-hidden';
                existingBadge.textContent = 'LOW HIT RATE';
            } else {
                var playerEl2 = el.querySelector('.compact-bet-player');
                if (playerEl2) {
                    var badge2 = document.createElement('span');
                    badge2.className = 'filter-badge filter-hidden';
                    badge2.textContent = 'LOW HIT RATE';
                    playerEl2.appendChild(badge2);
                }
            }
        } else if (existingBadge) {
            existingBadge.remove();
        }

        // Render live in-game stat tracker if game is in progress
        if (fresh.live_game) {
            var lg = fresh.live_game;
            var gs = lg.game_status || {};
            var clock = gs.time_remaining || gs.clock || gs.raw || '';
            var period = gs.period_label || gs.raw || 'Q' + (gs.period || '');
            var currentStat = lg.current_stat || 0;
            var projected = (lg.projected_final || 0).toFixed(1);
            var onPace = lg.on_pace;
            var progressPct = Math.min(lg.progress_pct || 0, 200);

            // Build compact inline live tracker — works for both game picks and hero picks
            var liveEl = el.querySelector('.live-inline-tracker');
            if (!liveEl) {
                liveEl = document.createElement('div');
                liveEl.className = 'live-inline-tracker';
                // Try multiple insertion points depending on element type
                var insertTarget =
                    el.querySelector('.compact-bet-prop') ||   // game pick cards
                    el.querySelector('.hero-prop') ||           // hero picks
                    el.querySelector('.compact-bet-player') ||  // fallback
                    el.querySelector('.hero-player');            // fallback
                if (insertTarget && insertTarget.parentNode) {
                    insertTarget.parentNode.insertBefore(liveEl, insertTarget.nextSibling);
                } else {
                    // Last resort: append to the element itself
                    el.appendChild(liveEl);
                }
            }

            var paceClass = onPace ? 'pace-on' : 'pace-off';
            var paceIcon = onPace ? 'ON PACE' : 'BEHIND';
            var barColor = progressPct >= 100 ? '#00e676' : (progressPct >= 70 ? '#ff9800' : 'var(--accent)');

            // Event badges
            var evtHtml = '';
            (lg.events || []).forEach(function(e) {
                if (e.type === 'foul_trouble') evtHtml += '<span class="live-evt foul">5 FOULS</span>';
                if (e.type === 'blowout') evtHtml += '<span class="live-evt blowout">BLOWOUT</span>';
                if (e.type === 'possible_injury') evtHtml += '<span class="live-evt injury">INJ?</span>';
            });

            liveEl.innerHTML =
                '<div class="live-inline-row">' +
                    '<span class="live-inline-badge">LIVE</span>' +
                    '<span class="live-inline-clock">' + period + ' ' + clock + '</span>' +
                    '<span class="live-inline-stat">Currently: <strong>' + currentStat + '</strong></span>' +
                    '<span class="live-inline-proj">Proj: ' + projected + '</span>' +
                    '<span class="live-inline-pace ' + paceClass + '">' + paceIcon + '</span>' +
                    evtHtml +
                '</div>' +
                '<div class="live-inline-bar">' +
                    '<div class="live-inline-fill" style="width:' + Math.min(progressPct, 100) + '%;background:' + barColor + '"></div>' +
                '</div>';

            // Flash the stat if it changed
            var statEl = liveEl.querySelector('.live-inline-stat strong');
            if (statEl && el.dataset.lastLiveStat !== String(currentStat)) {
                statEl.classList.remove('stat-bump');
                void statEl.offsetWidth;
                statEl.classList.add('stat-bump');
                el.dataset.lastLiveStat = String(currentStat);
            }
            // Check if this prop has resolved — ONLY on definitive conditions
            var gs2 = lg.game_status || {};
            var gameOver = gs2.is_final === true;
            var overAlreadyHit = currentStat > fresh.line;
            var side2 = fresh.side || el.dataset.side;

            // Only grey out when: game is final, OR over side definitively hit (stat exceeded line)
            if (gameOver) {
                var isHit = (side2 === 'over' && overAlreadyHit) || (side2 === 'under' && !overAlreadyHit);
                if (!el.classList.contains('pick-resolved')) {
                    el.classList.add('pick-resolved', isHit ? 'pick-hit' : 'pick-missed');
                }
            } else if (side2 === 'over' && overAlreadyHit) {
                // Over already hit mid-game — hide it, it's done
                el.style.display = 'none';
                return;
            } else if (side2 === 'under' && overAlreadyHit) {
                // Under is dead — stat already passed the line, hide it
                el.style.display = 'none';
                return;
            } else {
                el.style.display = '';
                el.classList.remove('pick-resolved', 'pick-hit', 'pick-missed');
            }
        } else {
            // Remove live tracker if game not in progress
            var oldTracker = el.querySelector('.live-inline-tracker');
            if (oldTracker) oldTracker.remove();
            el.classList.remove('pick-resolved', 'pick-hit', 'pick-missed');
        }
    });

    // Update hero picks with fresh edge values
    updateHeroPicksRealtime(freshLookup);

    // Also update the previousLines for stale detection
    for (var key in freshLookup) {
        var pred = freshLookup[key];
        previousLines[key] = {
            line: pred.line,
            over_odds: pred.over_odds,
            under_odds: pred.under_odds,
        };
    }
}

function renderLiveGameOverlay(el, liveData) {
    // Render live in-game stats overlay on a pick card
    if (!liveData || liveData.source !== 'v2_live_game') return;

    var existing = el.querySelector('.live-game-overlay');
    if (!existing) {
        existing = document.createElement('div');
        existing.className = 'live-game-overlay';
        el.insertBefore(existing, el.firstChild);
    }

    var currentStat = liveData.current_stat || 0;
    var line = liveData.line || 0;
    var projected = liveData.projected_final || 0;
    var progressPct = Math.min((liveData.progress_pct || 0), 200);
    var pOver = Math.round((liveData.live_p_over || 0.5) * 100);
    var minsPlayed = (liveData.minutes_played || 0).toFixed(0);
    var gameStatus = liveData.game_status || {};
    var quarter = gameStatus.period_label || gameStatus.raw || ('Q' + (gameStatus.period || ''));
    var clock = gameStatus.time_remaining || gameStatus.clock || '';
    var onPace = liveData.on_pace;

    var paceClass = onPace ? 'pace-on' : 'pace-off';
    var paceText = onPace ? 'ON PACE' : 'BEHIND';
    var barColor = progressPct >= 100 ? '#00e676' : (progressPct >= 75 ? '#ff9800' : 'var(--accent)');

    // Event badges
    var eventHtml = '';
    (liveData.events || []).forEach(function(evt) {
        var badge = '';
        if (evt.type === 'blowout') badge = '<span class="live-event-badge blowout">BLOWOUT ' + evt.margin + 'pt</span>';
        else if (evt.type === 'foul_trouble') badge = '<span class="live-event-badge foul">FOUL TROUBLE</span>';
        else if (evt.type === 'overtime_likely') badge = '<span class="live-event-badge ot">OT LIKELY</span>';
        else if (evt.type === 'possible_injury') badge = '<span class="live-event-badge injury">POSSIBLE INJURY</span>';
        if (badge) eventHtml += badge;
    });

    existing.innerHTML =
        '<div class="live-header">' +
            '<span class="live-badge">LIVE</span>' +
            '<span class="live-quarter">' + quarter + ' ' + clock + '</span>' +
            '<span class="live-mins">' + minsPlayed + ' min played</span>' +
            eventHtml +
        '</div>' +
        '<div class="live-stat-row">' +
            '<span class="live-current-stat mono">' + currentStat + '</span>' +
            '<span class="live-stat-label">current</span>' +
            '<span class="live-line mono">' + line.toFixed(1) + '</span>' +
            '<span class="live-stat-label">line</span>' +
            '<span class="live-projected mono">' + projected.toFixed(1) + '</span>' +
            '<span class="live-stat-label">projected</span>' +
        '</div>' +
        '<div class="live-progress-bar">' +
            '<div class="live-progress-fill" style="width:' + Math.min(progressPct, 100) + '%;background:' + barColor + '"></div>' +
            '<div class="live-progress-line" style="left:' + Math.min(100 / Math.max(line, 1) * line, 100) + '%"></div>' +
        '</div>' +
        '<div class="live-prob">' +
            '<span class="' + paceClass + '">' + paceText + '</span>' +
            '<span class="live-p-over">' + pOver + '% over</span>' +
        '</div>';
}

function updateHeroPicksRealtime(freshLookup) {
    // Update hero picks IN-PLACE with animations instead of full re-render
    if (typeof allTopPicks === 'undefined' || !allTopPicks.length) return;

    var changed = false;
    allTopPicks.forEach(function(pick) {
        var key = pick.player + '|' + pick.prop_type;
        var fresh = freshLookup[key];
        if (fresh) {
            pick.edge_percent = fresh.edge_percent;
            pick.edge = fresh.edge_percent;
            pick.our_probability = fresh.our_probability;
            pick.prob = fresh.our_probability;
            pick.over_odds = fresh.over_odds;
            pick.under_odds = fresh.under_odds;
            pick.dk_odds = fresh.dk_odds;
            pick.best_book = fresh.best_book;
            pick.best_odds = fresh.best_odds;
            pick.line = fresh.line;
            pick.all_books = fresh.all_books;
            pick.line_shopping = fresh.line_shopping;
            pick.odds_improvement = fresh.odds_improvement || 0;
            pick.odds_tier = fresh.odds_tier || pick.odds_tier || 'standard';
            pick.arbitrage = fresh.arbitrage || [];
            pick.middles = fresh.middles || [];
            pick.source = fresh.source;
            pick.live_game = fresh.live_game || null;
            pick.filter_tier = fresh.filter_tier || pick.filter_tier || 'marginal';
            pick.filter_reason = fresh.filter_reason || pick.filter_reason || '';
            // Consensus fields
            pick.soft_book_has_best = fresh.soft_book_has_best || false;
            pick.is_star_player = fresh.is_star_player || false;
            pick.line_range = fresh.line_range || 0;
            pick.consensus_over_prob = fresh.consensus_over_prob || null;
            pick.consensus_under_prob = fresh.consensus_under_prob || null;
            pick.consensus_edge = fresh.consensus_edge || null;
            pick.soft_book_score = fresh.soft_book_score || 0;
            pick.composite_score = fresh.composite_score || 0;
            changed = true;

            // Mark as resolved only on definitive conditions
            var lg = fresh.live_game;
            if (lg && lg.current_stat != null) {
                var gameIsFinal = lg.game_status && lg.game_status.is_final === true;
                var overHit = lg.current_stat > pick.line;
                // Only resolve: game final, or over bet already hit, or under bet already lost
                if (gameIsFinal || (pick.side === 'over' && overHit) || (pick.side === 'under' && overHit)) {
                    pick.resolved = true;
                } else {
                    pick.resolved = false;
                }
            }
        }
    });

    if (!changed) return;

    // Always re-render hero picks — live data is baked into the HTML via IIFE
    renderHeroPicks();
}

function updateLiveLines(data) {
    var games = data.games || [];

    // Build a lookup of current live lines keyed by player|prop_type
    var currentLines = {};
    games.forEach(function(game) {
        var players = game.players || {};
        for (var playerName in players) {
            var props = players[playerName];
            props.forEach(function(prop) {
                var key = playerName + '|' + prop.prop_type;
                currentLines[key] = {
                    line: prop.line,
                    over_odds: prop.over_odds,
                    under_odds: prop.under_odds,
                };
            });
        }
    });

    // Detect line movements vs previous scan (ticker display)
    for (var key in currentLines) {
        var prev = previousLines[key];
        var curr = currentLines[key];

        if (prev && prev.line !== curr.line) {
            highlightLineChange(key, prev.line, curr.line);
        }
    }

    previousLines = currentLines;

    // --- Stale pick detection ---
    // Compare live lines against the analyzed lines stored on pick elements
    var pickElements = document.querySelectorAll('[data-analyzed-line]');
    pickElements.forEach(function(el) {
        var player = el.dataset.player;
        var propType = el.dataset.propType;
        if (!player || !propType) return;

        var liveKey = player + '|' + propType;
        var liveData = currentLines[liveKey];
        if (!liveData) return;  // No live data for this pick

        var analyzedLine = parseFloat(el.dataset.analyzedLine);
        if (isNaN(analyzedLine)) return;

        if (liveData.line !== analyzedLine) {
            // Line moved from analyzed value -- mark as STALE
            markPickStale(el, {
                old_line: analyzedLine,
                new_line: liveData.line,
                direction: liveData.line > analyzedLine ? 'up' : 'down'
            });
        } else {
            // Line is same -- update odds display with flash animation
            clearStaleBadge(el);
            updatePickOdds(el, liveData.over_odds, liveData.under_odds);
        }
    });

    // --- Auto-refresh game scores ---
    updateLiveScores(games);
}


/* =====================================================
   STALE PICK DETECTION & LIVE ODDS UPDATE
   ===================================================== */

function markPickStale(el, movement) {
    // Don't re-mark if already stale with the same new_line
    if (el.classList.contains('pick-stale') && el.dataset.staleLine === String(movement.new_line)) {
        return;
    }

    el.classList.add('pick-stale');
    el.dataset.staleLine = String(movement.new_line);

    // Remove any existing stale badge
    var existingBadge = el.querySelector('.stale-badge');
    if (existingBadge) existingBadge.remove();

    // Build the stale badge HTML
    var arrow = movement.direction === 'up' ? '\u2191' : '\u2193';
    var gameId = el.dataset.gameId || '';

    // Try to find the game card this pick belongs to (for re-analyze)
    if (!gameId) {
        var gameCard = el.closest('.game-card');
        if (gameCard) {
            gameId = gameCard.dataset.gameId || gameCard.id.replace('game-card-', '');
        }
    }

    var reanalyzeBtn = '';
    if (gameId) {
        reanalyzeBtn = '<button class="btn-reanalyze" onclick="event.stopPropagation(); reanalyzeGame(\'' + gameId + '\', this)">Re-analyze</button>';
    }

    var badge = document.createElement('div');
    badge.className = 'stale-badge';
    badge.innerHTML =
        '<span class="stale-icon">\u26A0</span>' +
        '<span class="stale-text">LINE MOVED: ' +
            '<span class="mono">' + movement.old_line.toFixed(1) + '</span>' +
            ' \u2192 ' +
            '<span class="mono stale-new-line stale-' + movement.direction + '">' + movement.new_line.toFixed(1) + '</span>' +
            ' ' + arrow +
        '</span>' +
        reanalyzeBtn;

    // Insert badge at the top of the element
    el.insertBefore(badge, el.firstChild);
}

function clearStaleBadge(el) {
    if (!el.classList.contains('pick-stale')) return;
    el.classList.remove('pick-stale');
    delete el.dataset.staleLine;
    var badge = el.querySelector('.stale-badge');
    if (badge) badge.remove();
}

function updatePickOdds(el, overOdds, underOdds) {
    // Update the DK odds value display in-place with a brief flash
    var side = el.dataset.side;
    var relevantOdds = (side === 'under') ? underOdds : overOdds;
    if (relevantOdds == null) return;

    var oddsEl = el.querySelector('.dk-odds-value');
    if (!oddsEl) return;

    var currentText = oddsEl.textContent.trim();
    var newText = formatOdds(relevantOdds);

    if (currentText !== newText) {
        oddsEl.textContent = newText;
        oddsEl.classList.remove('odds-flash');
        void oddsEl.offsetWidth;
        oddsEl.classList.add('odds-flash');
    }
}

function updatePickOddsAnimated(el, overOdds, underOdds, side) {
    // Enhanced odds update with directional color (green = better for you, red = worse)
    var relevantOdds = (side === 'under') ? underOdds : overOdds;
    if (relevantOdds == null) return;

    // Update all odds elements (DK + cross-book rows)
    var allOddsEls = el.querySelectorAll('.dk-odds-value, .hero-odds-row .odds');
    allOddsEls.forEach(function(oddsEl) {
        var currentText = oddsEl.textContent.trim();
        var newText = formatOdds(relevantOdds);

        // Only flash DK odds element specifically
        if (!oddsEl.classList.contains('dk-odds-value')) return;

        if (currentText !== newText) {
            var oldVal = parseOdds(currentText);
            var newVal = relevantOdds;
            var improved = newVal > oldVal; // Higher odds = better for bettor

            oddsEl.textContent = newText;
            oddsEl.classList.remove('odds-flash', 'odds-up', 'odds-down');
            void oddsEl.offsetWidth;
            oddsEl.classList.add(improved ? 'odds-up' : 'odds-down');

            // Also briefly highlight the whole pick card
            el.classList.remove('pick-pulse');
            void el.offsetWidth;
            el.classList.add('pick-pulse');
        }
    });
}

function parseOdds(text) {
    // Parse "-110" or "+105" from text
    var cleaned = text.replace(/[^0-9\-\+]/g, '');
    return parseInt(cleaned, 10) || -110;
}

function reanalyzeGame(gameId, btnEl) {
    if (!gameId) return;

    // Show loading state on the button
    if (btnEl) {
        btnEl.disabled = true;
        btnEl.textContent = 'Analyzing...';
    }

    fetch('/api/game/' + gameId + '/reanalyze', { method: 'POST' })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) {
                console.warn('Re-analyze error:', data.error);
                if (btnEl) {
                    btnEl.disabled = false;
                    btnEl.textContent = 'Re-analyze';
                }
                return;
            }

            // Re-render the inline results for this game
            var container = document.getElementById('results-' + gameId);
            var card = document.getElementById('game-card-' + gameId);
            if (container) {
                renderInlineResults(container, data);
            }

            // Update hero picks if this game contributed
            collectTopPicks(data, parseInt(gameId, 10));

            // Update the analyze button
            if (card) {
                var analyzeBtn = card.querySelector('.btn-analyze');
                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = 'Re-Analyze';
                }
            }

            console.log('Re-analyzed game ' + gameId + ' with fresh lines');
        })
        .catch(function(err) {
            console.warn('Re-analyze failed:', err.message);
            if (btnEl) {
                btnEl.disabled = false;
                btnEl.textContent = 'Re-analyze';
            }
        });
}

function updateLiveScores(games) {
    // Update scores for live/final games from the live-lines response
    if (!games || !games.length) return;

    games.forEach(function(game) {
        // The live-lines endpoint returns event_id but game cards use game_id.
        // Try to match by home/away teams on the displayed cards.
        var cards = document.querySelectorAll('.game-card');
        cards.forEach(function(card) {
            var matchup = card.querySelector('.matchup');
            if (!matchup) return;
            var teams = matchup.textContent.trim();
            // Check if this card matches the game by team names
            var homeTeam = game.home_team || '';
            var awayTeam = game.away_team || '';
            if (!homeTeam || !awayTeam) return;

            // Match on abbreviation in the card header
            var awayEl = matchup.querySelector('.away-team');
            var homeEl = matchup.querySelector('.home-team');
            if (!awayEl || !homeEl) return;

            // Only update scores if the game data includes them
            if (game.home_score == null && game.away_score == null) return;

            var statusCat = game.status_category || '';
            if (statusCat !== 'live' && statusCat !== 'final') return;

            var scoreTicker = card.querySelector('.live-score-ticker');
            if (scoreTicker) {
                var statusLabel = statusCat === 'live' ? 'LIVE' : 'FINAL';
                var statusColor = statusCat === 'live' ? 'color:var(--accent)' : 'color:var(--text-muted)';
                scoreTicker.innerHTML =
                    '<span class="score-away">' + awayEl.textContent.trim() + ' ' + (game.away_score || 0) + '</span>' +
                    '<span class="score-sep">-</span>' +
                    '<span class="score-home">' + (game.home_score || 0) + ' ' + homeEl.textContent.trim() + '</span>' +
                    '<span class="score-status" style="' + statusColor + '">' + statusLabel + '</span>';
            }
        });
    });
}

function highlightLineChange(key, oldLine, newLine) {
    var parts = key.split('|');
    if (parts.length !== 2) return;
    var playerName = parts[0];
    var propType = parts[1];
    var direction = newLine > oldLine ? 'up' : 'down';
    var arrow = direction === 'up' ? '\u2191' : '\u2193';

    var betElements = document.querySelectorAll('.compact-bet');
    betElements.forEach(function(el) {
        var playerEl = el.querySelector('.compact-bet-player a');
        var propEl = el.querySelector('.compact-bet-prop');
        if (!playerEl || !propEl) return;

        var elPlayer = playerEl.textContent.trim();
        var elProp = propEl.textContent || '';

        if (elPlayer === playerName && elProp.indexOf(formatPropType(propType)) !== -1) {
            el.classList.remove('line-changed');
            void el.offsetWidth;
            el.classList.add('line-changed');

            var existingArrow = el.querySelector('.movement-arrow');
            if (existingArrow) {
                existingArrow.remove();
            }
            var arrowSpan = document.createElement('span');
            arrowSpan.className = 'movement-arrow movement-' + direction;
            arrowSpan.textContent = arrow + ' ' + oldLine.toFixed(1) + '\u2192' + newLine.toFixed(1);
            arrowSpan.title = 'Line moved from ' + oldLine.toFixed(1) + ' to ' + newLine.toFixed(1);
            var betInfo = el.querySelector('.compact-bet-prop');
            if (betInfo) {
                betInfo.appendChild(arrowSpan);
            }
        }
    });
}

function updateMovementTicker(movements) {
    var ticker = document.getElementById('live-ticker');
    var itemsContainer = document.getElementById('ticker-items');
    if (!ticker || !itemsContainer) return;

    var lineMoves = movements.filter(function(m) {
        return m.direction === 'up' || m.direction === 'down';
    });

    if (lineMoves.length === 0) {
        if (livePollingActive) {
            ticker.style.display = 'flex';
            itemsContainer.innerHTML = '<span class="ticker-empty">No line movements detected since last scan</span>';
        }
        return;
    }

    ticker.style.display = 'flex';
    itemsContainer.innerHTML = '';

    lineMoves.forEach(function(m) {
        var item = document.createElement('span');
        item.className = 'ticker-item';

        var arrow = m.direction === 'up' ? '\u2191' : '\u2193';
        var movClass = 'movement-' + m.direction;

        item.innerHTML =
            '<span class="ticker-item-player">' + m.player + '</span> ' +
            '<span class="ticker-item-prop">' + formatPropType(m.prop_type) + '</span> ' +
            '<span class="ticker-item-movement ' + movClass + ' mono">' +
                m.old_line.toFixed(1) + '\u2192' + m.new_line.toFixed(1) + ' ' + arrow +
            '</span>';

        itemsContainer.appendChild(item);
    });

    ticker.classList.remove('new-opportunity');
    void ticker.offsetWidth;
    ticker.classList.add('new-opportunity');
}

function updateCrossBookEdges(edges) {
    var edgeContainer = document.getElementById('ticker-cross-book');
    if (!edgeContainer) return;

    if (!edges || edges.length === 0) {
        edgeContainer.style.display = 'none';
        return;
    }

    edgeContainer.style.display = 'inline-flex';
    var bestEdge = edges[0];
    edgeContainer.innerHTML =
        '<span class="cross-book-edge-badge">' +
            bestEdge.player + ' ' + formatPropType(bestEdge.prop_type) +
            ' DK:' + bestEdge.dk_line.toFixed(1) +
            ' vs ' + bestEdge.other_book + ':' + bestEdge.other_line.toFixed(1) +
        '</span>';

    if (edges.length > 1) {
        edgeContainer.innerHTML += '<span class="cross-book-edge-badge">+' + (edges.length - 1) + ' more</span>';
    }
}

function updateLiveStatusIndicator(data) {
    // Show a persistent live status bar so user always knows the system is active
    var indicator = document.getElementById('live-status-indicator');
    if (!indicator) {
        // Create it if it doesn't exist — insert after navbar
        indicator = document.createElement('div');
        indicator.id = 'live-status-indicator';
        indicator.className = 'live-status-bar';
        var nav = document.querySelector('.navbar');
        if (nav && nav.parentNode) {
            nav.parentNode.insertBefore(indicator, nav.nextSibling);
        } else {
            document.body.prepend(indicator);
        }
    }

    var totalPreds = data.total_predictions || 0;
    var movements = (data.movements || []).length;
    var liveGames = data.live_games || 0;
    var recalcMs = data.recalc_time_ms || 0;

    var movementText = movements > 0
        ? '<span class="status-movements">' + movements + ' line move' + (movements > 1 ? 's' : '') + '</span>'
        : '<span class="status-stable">Odds stable</span>';

    var liveText = liveGames > 0
        ? ' <span class="status-live-games">' + liveGames + ' live game' + (liveGames > 1 ? 's' : '') + '</span>'
        : '';

    // Compute shopping stats from all displayed predictions
    var shoppingText = '';
    if (data.games) {
        var totalBetter = 0;
        var totalPicks = 0;
        var totalArbCount = 0;
        var totalMiddleCount = 0;
        var evGains = [];
        for (var gid in data.games) {
            var preds = (data.games[gid].predictions || []);
            preds.forEach(function(p) {
                totalPicks++;
                if (p.best_book && p.best_book !== 'draftkings' && p.best_odds > (p.dk_odds || -110)) {
                    totalBetter++;
                }
                if (p.odds_improvement > 0) evGains.push(p.odds_improvement);
                if (p.arbitrage && p.arbitrage.length > 0) totalArbCount += p.arbitrage.length;
                if (p.middles && p.middles.length > 0) totalMiddleCount += p.middles.length;
            });
        }
        if (totalBetter > 0) {
            var avgGain = evGains.length > 0 ? evGains.reduce(function(a,b){return a+b;},0) / evGains.length : 0;
            shoppingText = ' <span class="status-shopping">' + totalBetter + ' better prices' +
                (avgGain > 0.1 ? ' (+' + avgGain.toFixed(1) + '% avg EV)' : '') + '</span>';
        }
        if (totalArbCount > 0) {
            shoppingText += ' <span class="status-arb">&#9888; ' + totalArbCount + ' arb</span>';
        }
        if (totalMiddleCount > 0) {
            shoppingText += ' <span class="status-middle">' + totalMiddleCount + ' middle' + (totalMiddleCount > 1 ? 's' : '') + '</span>';
        }
    }

    indicator.innerHTML =
        '<span class="status-dot"></span>' +
        '<span class="status-text">Monitoring ' + totalPreds + ' props across all books</span>' +
        movementText + liveText + shoppingText +
        '<span class="status-recalc">' + recalcMs.toFixed(0) + 'ms</span>';

    // Pulse the dot on each update
    var dot = indicator.querySelector('.status-dot');
    if (dot) {
        dot.classList.remove('status-dot-pulse');
        void dot.offsetWidth;
        dot.classList.add('status-dot-pulse');
    }
}

function updateLastScanTime(timestamp) {
    var el = document.getElementById('ticker-time');
    if (!el) return;
    el.textContent = 'Just now';
    el.className = 'ticker-time';

    if (typeof updateFooterTimestamp === 'function') {
        updateFooterTimestamp();
    }
}

function updateScanAgoText() {
    if (!lastScanTimestamp) return;
    var el = document.getElementById('ticker-time');
    if (!el) return;

    var scanDate = new Date(lastScanTimestamp);
    var now = new Date();
    var diffMs = now - scanDate;
    var diffSec = Math.floor(diffMs / 1000);

    var text;
    var freshness;
    if (diffSec < 10) {
        text = 'Just now';
        freshness = 'fresh';
    } else if (diffSec < 60) {
        text = diffSec + 's ago';
        freshness = 'fresh';
    } else if (diffSec < 300) {
        var mins = Math.floor(diffSec / 60);
        text = mins + 'm ago';
        freshness = 'fresh';
    } else {
        var mins2 = Math.floor(diffSec / 60);
        text = mins2 + 'm ago';
        freshness = 'stale';
    }

    el.textContent = text;

    var badge = document.getElementById('last-scan-badge');
    if (badge) {
        badge.className = 'last-scan-badge ' + freshness;
    }
}


/* =====================================================
   LINE SIGNALS SUMMARY
   ===================================================== */

function renderLineSignalsSummary(data) {
    var signals = data.line_signals;
    if (!signals) return;

    var sharpCount = 0;
    var crossBookCount = 0;
    for (var key in signals) {
        var s = signals[key];
        if (s.sharp_signal) sharpCount++;
        if (s.cross_book_has_edge) crossBookCount++;
    }

    if (sharpCount === 0 && crossBookCount === 0) return;

    var container = document.getElementById('line-signals-summary');
    if (!container) return;

    container.innerHTML =
        '<div class="line-signals-summary">' +
            '<span class="line-signals-title">Line Signals:</span>' +
            (sharpCount > 0 ? ' <span class="line-signal-badge line-signal-sharp-agree">' + sharpCount + ' sharp</span>' : '') +
            (crossBookCount > 0 ? ' <span class="line-signal-badge line-signal-cross-book">' + crossBookCount + ' cross-book</span>' : '') +
        '</div>';

    container.style.display = 'block';
}


/* =====================================================
   INJURY CASCADE ALERTS
   ===================================================== */

var cascadeLoaded = false;

function loadCascadeAlerts() {
    fetch('/api/injury-cascade')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            renderCascadeAlerts(data);
            cascadeLoaded = true;
        })
        .catch(function(err) {
            console.debug('Cascade alerts not available:', err);
        });
}

function renderCascadeAlerts(data) {
    var section = document.getElementById('cascade-alerts-section');
    if (!section) return;

    var cascades = data.pregame_cascades || [];
    var alerts = data.alerts || [];

    // Merge: show alerts first, then pregame cascades
    var allItems = alerts.concat(cascades);

    if (allItems.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    var countEl = document.getElementById('cascade-count');
    var titleEl = document.getElementById('cascade-title');
    if (countEl) countEl.textContent = allItems.length;
    if (titleEl) {
        titleEl.textContent = allItems.length + ' Injury Cascade' + (allItems.length > 1 ? 's' : '') + ' Active';
    }

    var itemsEl = document.getElementById('cascade-items');
    if (!itemsEl) return;

    var html = '';
    allItems.forEach(function(item) {
        var absentPlayer = item.absent_player || '';
        var teamAbbrev = item.team_abbrev || '???';
        var status = item.status || item.status_change || 'Out';
        var comment = item.comment || '';
        var impacts = item.cascade_impacts || [];

        html += '<div class="cascade-item">';
        html += '<div class="cascade-item-header">';
        html += '<span class="cascade-player-out">' + escapeHtml(absentPlayer) + '</span>';
        html += '<span class="cascade-team-badge">' + escapeHtml(teamAbbrev) + '</span>';
        html += '<span class="cascade-status">' + escapeHtml(status) + '</span>';
        if (comment) {
            html += '<span style="font-size:0.65rem;color:var(--text-muted);margin-left:auto">' + escapeHtml(comment) + '</span>';
        }
        html += '</div>';

        if (impacts.length > 0) {
            html += '<div class="cascade-impacts">';
            impacts.forEach(function(imp) {
                var boost = imp.boost_pct || 0;
                var sign = boost > 0 ? '+' : '';
                var boostClass = boost > 0 ? 'boost-positive' : 'boost-negative';
                var confClass = imp.confidence === 'high' ? 'confidence-high' :
                               (imp.confidence === 'medium' ? 'confidence-medium' : '');
                html += '<span class="cascade-impact-chip ' + confClass + '">';
                html += '<span class="impact-player">' + escapeHtml(imp.player) + '</span>';
                html += '<span class="impact-prop">' + escapeHtml(imp.prop) + '</span>';
                html += '<span class="' + boostClass + '">' + sign + boost.toFixed(1) + '%</span>';
                html += '</span>';
            });
            html += '</div>';
        }

        html += '</div>';
    });

    itemsEl.innerHTML = html;
}

function toggleCascadeDetails() {
    var body = document.getElementById('cascade-body');
    var icon = document.getElementById('cascade-expand-icon');
    if (!body) return;
    var isOpen = body.style.display !== 'none';
    body.style.display = isOpen ? 'none' : 'block';
    if (icon) icon.classList.toggle('open', !isOpen);
}

// Load cascade alerts when page loads (non-blocking)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(loadCascadeAlerts, 2000);
    });
} else {
    setTimeout(loadCascadeAlerts, 2000);
}
