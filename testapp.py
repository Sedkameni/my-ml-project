import requests
import json

# Define the API endpoint URL (assuming local execution on port 5000)
API_URL = "http://localhost:5000/predict"


def test_prediction(tweet_text, expected_sentiment):
    """Sends a tweet to the API and checks the response."""
    print(f"\nTesting Tweet: '{tweet_text}'")

    # 1. Define the payload
    payload = {"tweet": tweet_text}

    # 2. Send the POST request
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes (4xx or 5xx)

        # 3. Parse the JSON response
        data = response.json()

        # 4. Check the prediction
        prediction = data.get('sentiment')

        if prediction == expected_sentiment:
            print(f"✅ PASS: Predicted '{prediction}' as expected.")
        else:
            print(f"❌ FAIL: Predicted '{prediction}', expected '{expected_sentiment}'.")

        print(f"   Cleaned Text: {data.get('cleaned_tweet')}")

    except requests.exceptions.RequestException as e:
        print(f"❌ FAIL: Could not connect to API or request failed: {e}")


if __name__ == "__main__":
    # Test cases
    test_prediction(
        tweet_text="The model training finally worked and the F1 score is excellent! #success",
        expected_sentiment="positive"
    )

    test_prediction(
        tweet_text="I hate when my code fails to compile. I'm having the worst day.",
        expected_sentiment="negative"
    )

    test_prediction(
        tweet_text="This is a neutral statement.",
        expected_sentiment="negative"
        # Neutral statements often default to the majority class or negative in binary models
    )