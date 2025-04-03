from pymongo import MongoClient
from datetime import datetime, timedelta
import logging

class MongoDB:
    def __init__(self):
        try:
            self.client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.server_info()
            self.db = self.client['urgent_detection']
            self.records = self.db['records']
            logging.info("Successfully connected to MongoDB")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def save_record(self, record):
        try:
            return self.records.insert_one(record)
        except Exception as e:
            logging.error(f"Error saving record: {e}")
            raise
    
    def get_user_history(self, user_id):
        try:
            return list(self.records.find({'user_id': user_id}).sort('timestamp', -1))
        except Exception as e:
            logging.error(f"Error getting user history: {e}")
            return []
    
    def get_all_urgent(self):
        return list(self.records.find({'is_urgent': True}).sort('timestamp', -1))
    
    def get_recent_detections(self, hours=24):
        time_threshold = datetime.now() - timedelta(hours=hours)
        return list(self.records.find({
            'timestamp': {'$gte': time_threshold}
        }).sort('timestamp', -1))
    
    def get_statistics(self):
        total = self.records.count_documents({})
        urgent = self.records.count_documents({'is_urgent': True})
        return {
            'total_detections': total,
            'urgent_cases': urgent,
            'normal_cases': total - urgent
        }