# Deployment Guide

## 1. Login to AWS Console

Start by logging into the AWS Management Console.

## 2. Create IAM User for Deployment

Create an IAM user with specific access permissions:

- **EC2 Access**: Allows access to virtual machines.
- **S3 Bucket**: Allows storing artifacts and models.
- **ECR (Elastic Container Registry)**: Allows saving Docker images in AWS.

### IAM Policy Attachments

Attach the following policies to the IAM user:

1. `AmazonEC2ContainerRegistryFullAccess`
2. `AmazonEC2FullAccess`
3. `AmazonS3FullAccess`

## 3. Create an S3 Bucket

- **Region**: ``
- **Bucket Name**: `

## 4. ECR Repository

Create an ECR repository to store/save Docker images:

- **ECR Repository URI**: ``

## 5. EC2 Machine Setup

Create an EC2 machine with Ubuntu.

## 6. Install Docker on EC2 Machine

Open your EC2 instance and install Docker:



Configure EC2 as Self-Hosted Runner

Configure EC2 as a self-hosted runner for GitHub Actions:

Go to Settings > Actions > Runners.
Click on New self-hosted runner.
Choose the operating system.
Follow the provided commands to complete the setup.

Setup GitHub Secrets

Add the following secrets to your GitHub repository:

Setup GitHub Secrets for the below fields.

AWS_ACCESS_KEY_ID: Your AWS access key.
AWS_SECRET_ACCESS_KEY: Your AWS secret access key.
AWS_REGION: 
AWS_ECR_LOGIN_URI: 
ECR_REPOSITORY_NAME:
MONGO_DB_URL:

Deployment Description
Follow these steps for deployment:

Build Docker Image: Build a Docker image of your source code.
Push Docker Image to ECR: Push the Docker image to the ECR repository.
Launch EC2 Instance: Start your EC2 instance.
Pull Docker Image on EC2: Pull the Docker image from ECR to your EC2 instance.
Launch Docker Image on EC2: Run your Docker image on the EC2 instance.