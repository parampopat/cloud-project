import os
import io
import boto3
import json
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences, np2csv



# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
vocabulary_length = 9013


def get_label(text):
    data = []
    data.append(text)
    one_hot_test_messages = one_hot_encode(data, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    lists = encoded_test_messages.tolist()
    json_str = json.dumps(lists)
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='text/csv',
                                      Body=json_str)
    try:
        body =  json.loads(response['Body'].read().decode())
        label = "Spam" if body['predicted_label'][0][0] == 1 else "Ham"
        prob = "%.2f" % round(body['predicted_probability'][0][0] * 100, 2)
    except:
        label = "Error Processing"
        prob = "00.00"
    return label, prob
    
    
def send_email(email_id, label, prob):
    pass


def lambda_handler(event, context):
    payload = event['data']
    # payload = "spam"
    
    label, prob = get_label(payload)
    send_email(email_id="abc@xyz.com", label=label, prob=prob)
    
    return {
        'statusCode': 200,
        'body': json.dumps({"label":label, "probability":prob})
    }

