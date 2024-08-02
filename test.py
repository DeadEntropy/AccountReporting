
import pandas as pd
import os
import cProfile, pstats

def main():
    # Code to profile
    from bkanalysis.ui import ui
    from bkanalysis.transforms.master_transform import Loader
    Loader.GRANULAR_NUTMEG = False
    df_trans = ui.load_transactions(True, include_market=True, ignore_overrides=True)
    df_values = ui.transactions_to_values(df_trans.copy())
    return df_trans, ui.compute_price(df_values.copy(), "USD")

if __name__ == "__main__":
    # Create a profiler object
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run the main function
    df_trans, df = main()
    
    # Stop profiling
    profiler.disable()

    # Dump the profiling results to a file
    profiler.dump_stats("profile_results_mkt_test.prof")

    df.to_csv('test_price.csv', index=False)
    df_trans.to_csv('test_transaction.csv', index=False)

    with open('baseline_price.csv', 'r') as t1, open('test_price.csv', 'r') as t2:
        fileone = t1.readlines()
        filetwo = t2.readlines()

    with open('update_price.csv', 'w') as outFile:
        for line in filetwo:
            if line not in fileone:
                outFile.write(line)
        
    with open('baseline_transaction.csv', 'r') as t1, open('test_transaction.csv', 'r') as t2:
        fileone = t1.readlines()
        filetwo = t2.readlines()

    with open('update_transaction.csv', 'w') as outFile:
        for line in filetwo:
            if line not in fileone:
                outFile.write(line)
    
    if os.stat("update_price.csv").st_size != 0:
        raise Exception('Impact!')