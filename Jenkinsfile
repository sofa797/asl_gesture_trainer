pipeline {
    agent any

    environment {
        venv = "${WORKSPACE}/venv"
    }

    stages {
        stage('setup') {
            steps {
                echo 'creating virtual environment and installing dependencies...'
                sh 'python -m venv $venv'
                sh '. $venv/bin/activate && pip install --upgrade pip'
                sh '. $venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('lint') {
            steps {
                echo 'checking code style...'
                sh '. $venv/bin/activate && flake8 . || echo "lint warnings exist"'
            }
        }

        stage('test') {
            steps {
                echo 'running tests...'
                sh '. $venv/bin/activate && pytest tests/ --junitxml=results.xml'
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
                sh '. $venv/bin/activate && python app.py || echo "app run skipped or failed"'
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