from pymongo import MongoClient
from datetime import datetime, timedelta
import logging

class MongoDB:
    def __init__(self, db_name='emergency_db', collection_name='audio_records', logger=None):
        """
        Inisialisasi koneksi MongoDB
        """
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.records = self.db[collection_name]
        self.logger = logger or logging.getLogger(__name__)

    def save_record(self, record):
        """
        Menyimpan record ke database
        """
        try:
            result = self.records.insert_one(record)
            self.logger.info(f"Record saved with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            self.logger.error(f"Error saving record: {str(e)}", exc_info=True)
            raise

    def get_user_history(self, user_id, limit=100):
        """
        Mendapatkan riwayat pengguna dengan pagination
        """
        try:
            return list(
                self.records.find({'user_id': user_id})
                .sort('timestamp', -1)
                .limit(limit)
            )
        except Exception as e:
            self.logger.error(f"Error getting user history: {str(e)}")
            return []

    def get_all_urgent(self, hours=24):
        """
        Mendapatkan semua kasus darurat dalam 24 jam terakhir
        """
        try:
            time_threshold = datetime.now() - timedelta(hours=hours)
            return list(
                self.records.find({
                    'is_urgent': True,
                    'timestamp': {'$gte': time_threshold}
                }).sort('timestamp', -1)
            )
        except Exception as e:
            self.logger.error(f"Error getting urgent cases: {str(e)}")
            return []

    def get_recent_detections(self, hours=24, limit=100):
        """
        Mendapatkan deteksi terakhir dengan filter waktu
        """
        try:
            time_threshold = datetime.now() - timedelta(hours=hours)
            return list(
                self.records.find({'timestamp': {'$gte': time_threshold}})
                .sort('timestamp', -1)
                .limit(limit)
            )
        except Exception as e:
            self.logger.error(f"Error getting recent detections: {str(e)}")
            return []

    def get_statistics(self, hours=24):
        """
        Mendapatkan statistik deteksi
        """
        try:
            time_threshold = datetime.now() - timedelta(hours=hours)
            total = self.records.count_documents({
                'timestamp': {'$gte': time_threshold}
            })
            urgent = self.records.count_documents({
                'is_urgent': True,
                'timestamp': {'$gte': time_threshold}
            })
            return {
                'total_detections': total,
                'urgent_cases': urgent,
                'normal_cases': total - urgent,
                'time_window_hours': hours
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {
                'total_detections': 0,
                'urgent_cases': 0,
                'normal_cases': 0,
                'time_window_hours': hours
            }

    def close_connection(self):
        """
        Menutup koneksi database
        """
        try:
            self.client.close()
            self.logger.info("MongoDB connection closed")
        except Exception as e:
            self.logger.error(f"Error closing connection: {str(e)}")