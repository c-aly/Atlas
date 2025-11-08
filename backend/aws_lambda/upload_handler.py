"""
AWS Lambda upload handler - REMOVED

This file previously contained AWS Lambda logic for uploading images to S3
and triggering an embedding processor. For local-only development this
project does not require AWS. The original functionality was removed to
avoid depending on boto3 / AWS credentials.

If you need to re-enable cloud uploads later, restore the original file
from your git history or implement an alternative that uploads to a
local storage or mock service.
"""

def lambda_handler(event, context):
    raise RuntimeError("AWS Lambda upload handler is disabled for local development.")

