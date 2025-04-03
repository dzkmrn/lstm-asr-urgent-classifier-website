from database import MongoDB
from datetime import datetime

def test_connection():
    try:
        db = MongoDB()
        print("MongoDB connection successful!")
        
        # Test insert
        test_record = {
            'user_id': 'test_user',
            'timestamp': datetime.now(),
            'is_urgent': False,
            'test': True
        }
        result = db.save_record(test_record)
        print("Test record inserted successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()