"""Update the results JSON with deep dive findings."""
import json

with open('outputs/model_experiments_L_P.json') as f:
    results = json.load(f)

results['deep_dive'] = {
    'low_cv_hot_under': {
        'description': 'Bet UNDER when target_trend > threshold and target_cv < threshold. Low-CV means predictable player who is running hot -> regression.',
        'test_set_only': True,
        'best_configs': [
            {
                'cv_thresh': 0.20, 'trend_thresh': 2,
                'n': 81, 'hit_rate': 0.642, 'roi': 0.205,
                'ci_95': [0.013, 0.398], 'p_value': 0.005,
                'note': 'Highest ROI but small sample'
            },
            {
                'cv_thresh': 0.25, 'trend_thresh': 2,
                'n': 220, 'hit_rate': 0.577, 'roi': 0.085,
                'ci_95': [-0.038, 0.204], 'p_value': 0.011,
                'note': 'Good balance of volume and signal'
            },
            {
                'cv_thresh': 0.30, 'trend_thresh': 2,
                'n': 387, 'hit_rate': 0.579, 'roi': 0.087,
                'ci_95': [-0.008, 0.178], 'p_value': 0.001,
                'note': 'Highest volume profitable config'
            },
            {
                'cv_thresh': 0.20, 'trend_thresh': 1,
                'n': 127, 'hit_rate': 0.622, 'roi': 0.166,
                'ci_95': [0.011, 0.327], 'p_value': 0.003,
                'note': 'More volume at lower trend threshold'
            },
        ],
        'validation_set_check': {
            'cv_0.20_trend_2': {'n': 72, 'hit_rate': 0.542, 'roi': 0.010},
            'cv_0.25_trend_2': {'n': 189, 'hit_rate': 0.524, 'roi': -0.025},
            'cv_0.30_trend_2': {'n': 387, 'hit_rate': 0.506, 'roi': -0.055},
            'WARNING': 'Validation set does NOT confirm the signal. ROI is flat or negative.',
        },
        'train_set_check': {
            'cv_0.25_trend_2': {'n': 1307, 'hit_rate': 0.553, 'roi': 0.045},
            'cv_0.30_trend_2': {'n': 2378, 'hit_rate': 0.549, 'roi': 0.037},
            'note': 'Train shows modest positive ROI, closer to validation than test.',
        },
        'by_prop_type': {
            'points': {'n': 18, 'hit_rate': 0.667, 'roi': 0.240, 'note': 'too few legs'},
            'pts_rebs_asts': {'n': 61, 'hit_rate': 0.623, 'roi': 0.170, 'note': 'dominant prop type'},
        },
        'seasonality': {
            'gp_40_plus': {'n': 367, 'hit_rate': 0.589, 'roi': 0.104, 'ci_95': [0.008, 0.198]},
            'gp_50_plus': {'n': 356, 'hit_rate': 0.581, 'roi': 0.091, 'ci_95': [-0.003, 0.189]},
            'gp_60_plus': {'n': 345, 'hit_rate': 0.586, 'roi': 0.099, 'ci_95': [0.001, 0.196]},
        },
    },
    'points_hot_streak_under': {
        'description': 'For points props specifically, bet UNDER when trend > 5 (extreme hot streak).',
        'n': 206, 'hit_rate': 0.568, 'roi': 0.072,
        'ci_95': [-0.059, 0.196],
        'trend_7': {'n': 72, 'hit_rate': 0.625, 'roi': 0.180, 'ci_95': [-0.034, 0.388]},
        'note': 'Points-specific; wider CI suggests noise component.',
    },
    'specialist_model_under': {
        'description': 'GBT specialist for points; bet UNDER when model p(over) < 0.42.',
        'points': {'n': 552, 'hit_rate': 0.571, 'roi': 0.055, 'ci_95': [-0.019, 0.131]},
        'pts_rebs_asts': {'n': 421, 'hit_rate': 0.553, 'roi': 0.028, 'ci_95': [-0.060, 0.116]},
    },
}

results['under_bias'] = {
    'description': 'Systematic UNDER bias across all splits and prop types.',
    'overall': {'train': 0.523, 'val': 0.527, 'test': 0.527},
    'by_prop_test': {
        'points': {'under_rate': 0.527, 'p_value': 0.003},
        'rebounds': {'under_rate': 0.515, 'p_value': 0.124},
        'assists': {'under_rate': 0.527, 'p_value': 0.011},
        'threes': {'under_rate': 0.535, 'p_value': 0.001},
        'pts_rebs_asts': {'under_rate': 0.533, 'p_value': 0.002},
    },
    'note': 'All UNDER signals benefit from this ~53% baseline. Must subtract baseline to isolate true signal.',
}

results['conclusions'] = {
    'profitable_signals': {
        '1_low_cv_hot_under': {
            'signal': 'cv < 0.30 AND trend > 2 -> UNDER',
            'test_roi': '+8.7%',
            'test_hit': '57.9%',
            'test_n': 387,
            'p_value': 0.001,
            'concern': 'Validation set does NOT confirm. Train shows +3.7%. Test ROI may be anomalously high.',
            'verdict': 'PROMISING BUT UNCONFIRMED. Need more out-of-sample data.',
        },
        '2_points_extreme_hot': {
            'signal': 'points prop AND trend > 5 -> UNDER',
            'test_roi': '+7.2%',
            'test_hit': '56.8%',
            'test_n': 206,
            'concern': 'CI includes zero [-5.9%, +19.6%]. Not statistically significant.',
            'verdict': 'SUGGESTIVE. Worth tracking but not deployable.',
        },
    },
    'not_profitable': {
        'fade_the_public': 'Juice gap contrarian signal loses money at every threshold.',
        'injury_cascade': 'Market already prices injuries. No exploitable edge.',
        'specialist_ensemble': 'Specialists mostly underperform global model. Less data = more noise.',
        'per_100_possessions': 'No improvement. Model already captures pace effects.',
        'cold_player_over': 'Market adjusts for cold streaks. No edge.',
    },
    'key_insight': (
        'The UNDER bias (~53% across all data) is the biggest structural edge. '
        'Low-CV hot player regression amplifies this to ~58%, but the signal '
        'is not confirmed in validation. The safest bet is still a blanket slight-UNDER lean, '
        'but that only yields ~-1.8% ROI (vig eats the edge). '
        'No approach tested here reliably beats the vig on test data with statistical significance.'
    ),
}

with open('outputs/model_experiments_L_P.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results updated: outputs/model_experiments_L_P.json")
