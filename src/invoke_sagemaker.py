import os
import io
import boto3
import json
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences, np2csv
import email




# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
vocabulary_length = 9013
client_email = boto3.client('ses', region_name="us-east-1")


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
        raw_prob = body['predicted_probability'][0][0]
        prob = raw_prob if body['predicted_label'][0][0] == 1 else 1 - raw_prob
        prob = "%.2f" % round( prob * 100, 2 )
    except:
        label = "Error Processing"
        prob = "00.00"
    return label, prob
    
    
def send_email(e, label, prob):
    
    lenbody = len(e['Body'])
    
    body = e['Body'][:240] + '...(truncated)' if lenbody > 240 else e['Body']
    
    msg = """
    We received your email sent at {sent} with the
    subject {subject}.
    Here is a 240 character sample of the email body:<br>
    {body}<br>
    The email was categorized as {label} with a
    {prob}% confidence.<br>
    Yours Truly, <br>
    Spamthusiasts
    """.format(
        sent = e['Date'], 
        subject=e['Subject'], 
        body=body,
        label=label,
        prob=prob
    )
    
    client_email.send_email(
        Source="detector@kampspamdetector.tech",
        Destination={'ToAddresses': [e['From']]},
        Message={
            'Subject': {
                'Data': 'Spam Detection Result',
                'Charset': 'UTF-8'
            },
            'Body': {
                'Html': {
                    'Data': msg,
                    'Charset': 'UTF-8'
                }
            }
        }
    )
    

def parse_email(e):
    parsed = email.message_from_string(e['Body'].read().decode())
    
    metakeys = ['From', 'Date', 'Subject'] 
    content = {k:v for k,v in parsed.items() if k in metakeys}
    content['Body'] = parsed.get_payload()[0].get_payload().replace('\r\n',' ' )

    
    return content


def lambda_handler(event, context):
    bucket=event['Records'][0]['s3']['bucket']['name']
    file=event['Records'][0]['s3']['object']['key']
    s3_client = boto3.client('s3')
    
    raw_email = s3_client.get_object(Bucket=bucket, Key=file)
    print("Decoding Email**********\n")
    email_content = parse_email(raw_email)
    print(email_content)
    print("Predicting**********\n")
    label, prob = get_label(email_content['Body'])
    print("Sending Email**********\n")
    
    send_email(email_content, label=label, prob=prob)
    
    return {
        'statusCode': 200,
        'body': json.dumps({"label":label, "probability":prob})
    }

