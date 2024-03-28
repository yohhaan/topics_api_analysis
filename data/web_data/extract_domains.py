import argparse
import os
import pandas as pd


if __name__ == "__main__":
    # Create Argument Parser
    parser = argparse.ArgumentParser(
        prog="python3 extract_domains.py",
        description="Extract domains from web raw data.",
    )
    parser.add_argument("browsing_csv_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    if not (os.path.isfile(args.browsing_csv_path)):
        raise Exception("Error: file missing")
    else:
        df = pd.read_csv(args.browsing_csv_path, sep=",")
        df["FQDN"] = df["subdomain"].fillna("") + df["domain"]
        pd.DataFrame(df["FQDN"].unique()).to_csv(
            args.output_path, index=False, header=False
        )
