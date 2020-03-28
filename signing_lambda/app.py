# noinspection PyUnresolvedReferences
import boto3
# noinspection PyUnresolvedReferences
from botocore.exceptions import ClientError
import os
import time
import json

s3_client = boto3.client('s3')


def handler(event, context):
    key = str(time.time())
    bucket = os.environ["BUCKET_NAME"]

    print(f"Key: {key} Bucket: {bucket}")

    try:
        signed_url = s3_client.generate_presigned_url('get_object',
                                                      Params={'Bucket': bucket,
                                                              'Key': key},
                                                      ExpiresIn=300)
    except ClientError as e:
        print("Error generating presigned url")
        print(e)
        return {
            "statusCode": 500,
            "errorType": "InternalServerError"
        }

    return {
        "statusCode": 200,
        "body": json.dumps({
            "signedUrl": signed_url
        })
    }
