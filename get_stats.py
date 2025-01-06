import pstats

# Load the profiling results
profile_path = "profile_results_mkt_baseline.prof"  # Replace with your profiling file path
p = pstats.Stats(profile_path)

# Sort the statistics by cumulative time and print the top 10 functions
p.sort_stats("cumulative").print_stats(60)
