# v2 - Line-anchor ML prop betting model
# Architecture: predicted = DK_line + ML_adjustment(features)
# LightGBM regressors that predict adjustments to DK lines.
# Target = actual - DK_line (how wrong was the line?)
