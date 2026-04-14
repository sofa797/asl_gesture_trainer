pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-u root:root'
        }
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONUNBUFFERED = '1'
    }

    stages {

        stage('install system dependencies') {
            steps {
                sh '''
                    apt-get update
                    apt-get install -y \
                        libgl1 \
                        libglib2.0-0 \
                        libsm6 \
                        libxext6 \
                        libxrender1
                '''
            }
        }

        stage('install python dependencies') {
            steps {
                sh '''
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install --no-cache-dir "keras>=3.0.0" --upgrade
                    pip install flake8 pytest pytest-cov
                '''
            }
        }

        stage('lint') {
            steps {
                sh '''
                    flake8 . || true
                '''
            }
        }

        stage('run tests') {
            steps {
                sh '''
                    pytest \
                        --cov=services \
                        --cov=utils \
                        --cov=flask_app \
                        --cov-report=term-missing \
                        --cov-report=html \
                        --junitxml=results.xml \
                        tests/
                '''
            }
        }

        stage('archive test results') {
            steps {
                junit 'results.xml'
                archiveArtifacts artifacts: 'htmlcov/**', fingerprint: true
            }
        }

        stage('archive artifacts') {
            steps {
                archiveArtifacts artifacts: 'asl_model.h5, logs/**', fingerprint: true
            }
        }
    }

    post {
        always {
            echo 'build finished'
        }
        success {
            echo 'success'
        }
        failure {
            echo 'failed'
        }
    }
}
