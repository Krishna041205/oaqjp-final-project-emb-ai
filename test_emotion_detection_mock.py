import unittest
from unittest.mock import patch
import sys

sys.path.append('.')

class MockResponse:
    def __init__(self, json_data, status_code):
        self.text = json_data
        self.status_code = status_code

def mocked_requests_post(*args, **kwargs):
    text = kwargs['json']['raw_document']['text']
    
    import json
    if text == 'I am glad this happened':
        emotions = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 1.0, 'sadness': 0.0}
    elif text == 'I am really mad about this':
        emotions = {'anger': 1.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
    elif text == 'I feel disgusted just hearing about this':
        emotions = {'anger': 0.0, 'disgust': 1.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
    elif text == 'I am so sad about this':
        emotions = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 1.0}
    elif text == 'I am really afraid that this will happen':
        emotions = {'anger': 0.0, 'disgust': 0.0, 'fear': 1.0, 'joy': 0.0, 'sadness': 0.0}
    else:
        emotions = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
        
    return MockResponse(json.dumps({'emotionPredictions': [{'emotion': emotions}]}), 200)

class TestEmotionDetector(unittest.TestCase):
    @patch('EmotionDetection.emotion_detection.requests.post', side_effect=mocked_requests_post)
    def test_emotion_detector(self, mock_post):
        from EmotionDetection.emotion_detection import emotion_detector
        
        # Test 1
        result_1 = emotion_detector('I am glad this happened')
        self.assertEqual(result_1['dominant_emotion'], 'joy')
        
        # Test 2
        result_2 = emotion_detector('I am really mad about this')
        self.assertEqual(result_2['dominant_emotion'], 'anger')
        
        # Test 3
        result_3 = emotion_detector('I feel disgusted just hearing about this')
        self.assertEqual(result_3['dominant_emotion'], 'disgust')
        
        # Test 4
        result_4 = emotion_detector('I am so sad about this')
        self.assertEqual(result_4['dominant_emotion'], 'sadness')
        
        # Test 5
        result_5 = emotion_detector('I am really afraid that this will happen')
        self.assertEqual(result_5['dominant_emotion'], 'fear')

if __name__ == '__main__':
    unittest.main()
