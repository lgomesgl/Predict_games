# No full
Model_RandomForestClassifier() :63.79% accuracy with std: 0.1
Model_XGBClassifier() :61.37% accuracy with std: 0.12
Model_SVC() :72.54% accuracy with std: 0.01

# No full + W/L%
Model_RandomForestClassifier() :66.04% accuracy with std: 0.08
Model_XGBClassifier() :61.7% accuracy with std: 0.12
Model_SVC() :72.7% accuracy with std: 0.01

# No full + GB
Model_RandomForestClassifier() :64.62% accuracy with std: 0.1
Model_XGBClassifier() :60.7% accuracy with std: 0.12
Model_SVC() :72.37% accuracy with std: 0.01

# No full + SRS
Model_RandomForestClassifier() :66.87% accuracy with std: 0.07
Model_XGBClassifier() :62.7% accuracy with std: 0.1
Model_SVC() :72.54% accuracy with std: 0.01

# No full + W/L% + GB
Model_RandomForestClassifier() :64.28% accuracy with std: 0.08
Model_XGBClassifier() :60.78% accuracy with std: 0.12
Model_SVC() :72.54% accuracy with std: 0.01

# No full + W/L% + GB + SRS
Model_RandomForestClassifier() :65.87% accuracy with std: 0.07
Model_XGBClassifier() :61.45% accuracy with std: 0.11
Model_SVC() :72.37% accuracy with std: 0.02

# Full
Model_RandomForestClassifier() :69.12% accuracy with std: 0.03
Model_XGBClassifier() :61.53% accuracy with std: 0.12
Model_SVC() :72.62% accuracy with std: 0.01

0                           RandomForestClassifier()  74.12% std: 0.03  0.636 std: 0.02
1  XGBClassifier(base_score=None, booster=None, c...  69.96% std: 0.05  0.609 std: 0.03
2  SVC(class_weight='balanced', kernel='linear', ...   55.86% std: 0.1   0.63 std: 0.03
3                                      BernoulliNB()  68.12% std: 0.02  0.626 std: 0.02