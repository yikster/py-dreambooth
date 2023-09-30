import json
import logging
import os
from typing import Optional, List
import boto3
from .misc import log_or_print


def create_bucket_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    s3_client = boto_session.client("s3")
    sts_client = boto_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    bucket_name = f"sagemaker-{region_name}-{account_id}"
    if (
        s3_client.head_bucket(Bucket=bucket_name)["ResponseMetadata"]["HTTPStatusCode"]
        == 404
    ):
        s3_client.create_bucket(Bucket=bucket_name)
        msg = f"The following S3 bucket was created: {bucket_name}"
        log_or_print(msg, logger)

    else:
        msg = f"The following S3 bucket was found: {bucket_name}"
        log_or_print(msg, logger)

    return bucket_name


def create_role_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    iam_client = boto_session.client("iam")

    role_name = f"AmazonSageMaker-ExecutionRole-{region_name}"
    try:
        role = iam_client.get_role(RoleName=role_name)
        msg = f"The following IAM role was found: {role['Role']['Arn']}"

    except iam_client.exceptions.NoSuchEntityException:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="SageMaker Execution Role",
        )
        policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        msg = f"The following IAM role was created: {role['Role']['Arn']}"

    log_or_print(msg, logger)
    return role_name


def delete_files_in_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_resource = boto_session.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=prefix):
        s3_resource.Object(bucket_name, obj.key).delete()
        msg = f"The 's3://{bucket_name}/{obj.key}' file has been deleted."
        log_or_print(msg, logger)


def make_s3_uri(bucket: str, prefix: str, filename: Optional[str] = None) -> str:
    prefix = prefix if filename is None else os.path.join(prefix, filename)
    return f"s3://{bucket}/{prefix}"


def upload_dir_to_s3(
    boto_session: boto3.Session,
    local_dir: str,
    bucket_name: str,
    prefix: str,
    file_ext_to_excl: Optional[List[str]] = None,
    public_readable: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")
    file_ext_to_excl = [] if file_ext_to_excl is None else file_ext_to_excl
    extra_args = {"ACL": "public-read"} if public_readable else {}

    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.split(".")[-1] not in file_ext_to_excl:
                s3_client.upload_file(
                    os.path.join(root, file),
                    bucket_name,
                    f"{prefix}/{file}",
                    ExtraArgs=extra_args,
                )
                msg = f"The '{file}' file has been uploaded to 's3://{bucket_name}/{prefix}/{file}'."
                log_or_print(msg, logger)
