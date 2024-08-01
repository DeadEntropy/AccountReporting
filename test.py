
import pandas as pd
import os
import cProfile, pstats

def main():
    # Code to profile
    from bkanalysis.ui import ui
    df_trans = ui.load_transactions(True, include_market=True, ignore_overrides=True)
    df_values = ui.transactions_to_values(df_trans.copy())
    return ui.compute_price(df_values.copy(), "USD")

if __name__ == "__main__":
    # Create a profiler object
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run the main function
    df = main()
    
    # Stop profiling
    profiler.disable()

    # Dump the profiling results to a file
    profiler.dump_stats("profile_results_mkt_baseline.prof")

    df.to_csv('baseline_mkt.csv', index=False)

    with open('baseline_mkt.csv', 'r') as t1, open('test_mkt.csv', 'r') as t2:
        fileone = t1.readlines()
        filetwo = t2.readlines()

    with open('update_mkt.csv', 'w') as outFile:
        for line in filetwo:
            if line not in fileone:
                outFile.write(line)
    
    if os.stat("update_mkt.csv").st_size != 0:
        raise Exception('Impact!')