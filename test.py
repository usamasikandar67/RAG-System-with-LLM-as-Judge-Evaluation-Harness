import json
import random
import subprocess
from datetime import datetime, timedelta

# Variables for configuration
START_DATE = "2023-02-01"  # Start date (format: YYYY-MM-DD)
END_DATE = "2023-09-30"    # End date (format: YYYY-MM-DD)
MAX_COMMITS_PER_DAY = 6    # Max commits per day
MIN_COMMITS_PER_DAY = 0    # Min commits per day
FILE_PATH = "./data.json"  # JSON file to store commit data
TXT_FILE_PATH = "./commit_dates.txt"  # Text file to store commit dates

# Convert start and end date to datetime objects
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

def make_commit(n):
    if n == 0:
        subprocess.run(["git", "push"])
        return

    # Randomly pick a commit date within the specified range
    days_range = (end_date - start_date).days
    random_days_offset = random.randint(0, days_range)
    commit_date = start_date + timedelta(days=random_days_offset)

    # Randomly decide how many commits to make on this day
    commits_today = random.randint(MIN_COMMITS_PER_DAY, MAX_COMMITS_PER_DAY)

    for _ in range(commits_today):
        date_str = commit_date.isoformat()

        data = {"date": date_str}
        print(date_str)

        # Write commit date to JSON file
        with open(FILE_PATH, "w") as f:
            json.dump(data, f)

        # Write commit date to text file
        with open(TXT_FILE_PATH, "a") as f_txt:
            f_txt.write(date_str + "\n")

        # Add and commit to Git
        subprocess.run(["git", "add", FILE_PATH])
        subprocess.run(["git", "commit", "--date", date_str, "-m", date_str])

    make_commit(n - 1)

# Start making commits
make_commit(80)