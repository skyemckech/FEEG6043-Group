import pickle
import os
from collections import defaultdict

class ScanDataManager:
    def __init__(self, storage_file='scan_data.pkl'):
        self.storage_file = storage_file
        self.labeled_scans = defaultdict(list)
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'rb') as f:
                self.labeled_scans = pickle.load(f)

    def save_scan(self, scan_data, label, source_file):
        key = f"{label}_{os.path.basename(source_file)}"
        self.labeled_scans[key].append(scan_data)
        self._save_data()

    def _save_data(self):
        with open(self.storage_file, 'wb') as f:
            pickle.dump(self.labeled_scans, f)

    def get_scans_by_label(self, label_pattern):
        return {k: v for k, v in self.labeled_scans.items() if label_pattern in k}

# Singleton instance
data_manager = ScanDataManager()