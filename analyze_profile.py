
import pstats
import argparse

def analyze(profile_file):
    p = pstats.Stats(profile_file)
    p.strip_dirs()
    
    print("\n=== Top 10 by Cumulative Time ===")
    p.sort_stats('cumulative').print_stats(10)
    
    print("\n=== Top 10 by Internal Time ===")
    p.sort_stats('tottime').print_stats(10)

    print("\n=== Detailed Analysis: Inference ===")
    p.print_callers("predict")
    p.print_callees("predict")
    
    print("\n=== Detailed Analysis: Sync Overhead ===")
    p.print_callers("_xpu_synchronize")
    
    print("\n=== Detailed Analysis: MCTS Expansion ===")
    p.print_callers("expand")
    p.print_callees("expand")

if __name__ == "__main__":
    analyze("profile.stats")
