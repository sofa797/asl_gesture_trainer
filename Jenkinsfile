pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-u root:root'
        }
    }

    stages {
        stage('setup') {
            steps {
                echo 'installing dependencies...'
                sh 'python -m pip install --upgrade pip'
                sh 'python -m pip install -r requirements.txt'
                sh 'python -m pip install flake8 pytest'
            }
        }

        stage('lint') {
            steps {
                echo 'checking code style...'
                sh 'flake8 . || echo "lint warnings exist"'
            }
        }

        stage('test') {
            steps {
                echo 'running tests...'
                sh 'pytest tests/ --junitxml=results.xml'
            }
            post {
                always {
                    junit 'results.xml'
                }
            }
        }

        stage('run app (optional)') {
            steps {
                echo 'running the application...'
                sh 'python app.py || echo "app run skipped or failed"'
            }
        }

        stage('archive artifacts') {
            steps {
                echo 'saving model and other artifacts...'
                archiveArtifacts artifacts: 'asl_model.h5, logs/**', fingerprint: true
            }
        }
    }

    post {
        always {
            echo 'build finished'
        }
        success {
            echo 'build completed successfully!'
        }
        failure {
            echo 'build failed!'
        }
    }
}