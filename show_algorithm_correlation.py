import pandas

file_name = 'LIVE_md_2_compared_results.csv'
result_comparison = pandas.read_csv(file_name)
print(file_name)
print(result_comparison.head())
print(result_comparison.mean())
# LIVE MD
print("Correlation:", result_comparison['DMOS score'].corr(result_comparison['im_quality_score']))
print(result_comparison.describe(include='all'))
print("------------------------------")