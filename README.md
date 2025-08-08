# InsurePredict: Interactive MLOps for Vehicle Insurance Forecasting

Welcome to InsurePredict, an interactive MLOps project designed to provide a robust and user-friendly pipeline for forecasting customer interest in vehicle insurance. This project showcases tools, techniques, services, and features for building, training, and deploying a machine learning model with real-time prediction capabilities. Follow along to explore project setup, data processing, interactive model training, and CI/CD automation, all accessible through a web-based interface!

## 📁 Project Setup and Structure

### Step 1: Project Template
Execute `template.py` to create the initial project template, including the required folder structure and placeholder files.

### Step 2: Package Management
Set up importing local packages in `setup.py` and `pyproject.toml` files.
> **Tip**: Learn more about these files from `crashcourse.txt`.

### Step 3: Virtual Environment and Dependencies
Create a virtual environment and install dependencies from `requirements.txt`:
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```
Verify the local packages by running:
```bash
pip list
```

### Step 4: Dataset Description
The dataset (`data.csv`) contains vehicle insurance customer data with the following columns:
- **id**: Unique ID for the customer
- **Gender**: Gender of the customer (Male/Female)
- **Age**: Age of the customer
- **Driving_License**: 0 (No DL), 1 (Has DL)
- **Region_Code**: Unique code for the customer's region
- **Previously_Insured**: 1 (Has vehicle insurance), 0 (No vehicle insurance)
- **Vehicle_Age**: Age of the vehicle (e.g., < 1 Year, 1-2 Years, > 2 Years)
- **Vehicle_Damage**: 1 (Vehicle damaged in the past), 0 (No damage)
- **Annual_Premium**: Yearly premium amount
- **Policy_Sales_Channel**: Anonymized code for outreach channel (e.g., Agents, Mail, Phone)
- **Vintage**: Number of days the customer has been associated with the company
- **Response**: 1 (Customer is interested), 0 (Customer is not interested)

## 📊 MongoDB Setup and Data Management

### Step 5: MongoDB Atlas Configuration
- Sign up for MongoDB Atlas and create a new project.
- Set up a free M0 cluster, configure the username and password, and allow access from any IP address (`0.0.0.0/0`).
- Retrieve the MongoDB connection string for Python and save it (replace `<password>` with your password).

### Step 6: Pushing Data to MongoDB
- Create a `notebook` folder, add the dataset (`data.csv`), and create a notebook file `mongoDB_demo.ipynb`.
- Use the notebook to push data to the MongoDB database.
- Verify the data in MongoDB Atlas under *Database > Browse Collections*.

## 📝 Logging, Exception Handling, and EDA

### Step 7: Set Up Logging and Exception Handling
Create logging and exception handling modules. Test them on a demo file `demo.py`.

### Step 8: Exploratory Data Analysis (EDA) and Feature Engineering
Analyze and engineer features in the *EDA and Feature Engg* notebook. Key EDA steps include:
- **Dataset Overview**: Check dataset shape, null values, and data types using `df.shape`, `df.isnull().sum()`, and `df.info()`.
- **Target Distribution**: Visualize the distribution of the `Response` column using a bar plot, showing the count of interested (1) vs. non-interested (0) customers.
- **Age Distribution**: Plot a histogram of `Age` to understand its distribution.
- **Age vs. Annual Premium**: Create a scatter plot to explore the relationship between `Age` and `Annual_Premium`.
- **Gender Analysis**: Compare `Gender` distribution and its relationship with `Response` and `Driving_License` using bar and categorical plots.
- **Vehicle-Related Features**: Analyze `Vehicle_Age`, `Vehicle_Damage`, and `Previously_Insured` distributions and their relationships with `Response` using count plots and categorical plots.
- **Premium Analysis**: Examine `Annual_Premium` statistics and distribution, identifying outliers (e.g., premiums > 200,000).

Feature engineering includes creating dummy variables for categorical features and scaling numerical features for model training.

## 📥 Data Ingestion

### Step 9: Data Ingestion Pipeline
- Define MongoDB connection functions in `configuration.mongo_db_connections.py`.
- Develop data ingestion components in `data_access` and `components.data_ingestion.py` to fetch and transform data.
- Update `entity/config_entity.py` and `entity/artifact_entity.py` with relevant ingestion configurations.
- Run `demo.py` after setting up MongoDB connection as an environment variable.

**Setting Environment Variables**  
Set MongoDB URL:
```bash
# For Bash
export MONGODB_URL="mongodb+srv://<username>:<password>..."
```
```powershell
# For PowerShell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>..."
```
> **Note**: On Windows, you can also set environment variables through system settings.

## 🔍 Data Validation, Transformation & Model Training

### Step 10: Data Validation
Define schema in `config.schema.yaml` and implement data validation functions in `utils.main_utils.py`.

### Step 11: Data Transformation
Implement data transformation logic in `components.data_transformation.py` and create `estimator.py` in the `entity` folder. Preprocessing steps include:
- **Categorical Encoding**: Map `Gender` to binary values (Female: 0, Male: 1) and create dummy variables for categorical features (`Vehicle_Age`, `Vehicle_Damage`, `Region_Code`, `Policy_Sales_Channel`).
- **Scaling**: Apply `StandardScaler` to numerical features (`Age`, `Vintage`) and `MinMaxScaler` to `Annual_Premium`.
- **Feature Selection**: Drop the `id` column as it’s not relevant for modeling.
- **Train-Test Split**: Split data into training and testing sets using `train_test_split` with a random state for reproducibility.

### Step 12: Model Training
Train a Random Forest Classifier using `RandomizedSearchCV` for hyperparameter tuning in `components.model_trainer.py`. Key steps include:
- Define hyperparameter grid for `criterion`, `max_depth`, `min_samples_leaf`, `min_samples_split`, and `n_estimators`.
- Fit the model on the training data (`x_train`, `y_train`) with 4-fold cross-validation.
- Save the trained model as `rf_model.pkl` using `pickle` for deployment.
- Load the model for predictions and evaluate performance using `classification_report`.

## 🌐 AWS Setup for Model Evaluation & Deployment

### Step 13: AWS Setup
- Log in to the AWS console, create an IAM user, and grant `AdministratorAccess`.
- Set AWS credentials as environment variables:
```bash
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
```
- Configure an S3 bucket and add access keys in `constants.__init__.py`.

### Step 14: Model Evaluation and Pushing to S3
- Create an S3 bucket named `my-model-mlopsproj` in the `us-east-1` region.
- Develop code to push/pull models to/from the S3 bucket in `src.aws_storage` and `entity/s3_estimator.py`.

## 🚀 Model Evaluation, Model Pusher, and Prediction Pipeline

### Step 15: Model Evaluation & Model Pusher
Implement model evaluation and deployment components. Evaluate the Random Forest model using metrics like precision, recall, and F1-score from `classification_report`. Create a Prediction Pipeline and set up `app.py` for API integration.

### Step 16: Static and Template Directory
Add `static` and `template` directories for web UI.

## 🔄 CI/CD Setup with Docker, GitHub Actions, and AWS

### Step 17: Docker and GitHub Actions
- Create `Dockerfile` and `.dockerignore`.
- Set up GitHub Actions with AWS authentication by creating secrets in GitHub for:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`
  - `ECR_REPO`

### Step 18: AWS EC2 and ECR
- Set up an EC2 instance for deployment.
- Install Docker on the EC2 machine.
- Connect EC2 as a self-hosted runner on GitHub.

### Step 19: Final Steps
- Open port `5080` on the EC2 instance.
- Access the deployed app by visiting `http://<public_ip>:5080`.

## 🛠️ Additional Resources
- **Crash Course on setup.py and pyproject.toml**: See `crashcourse.txt` for details.
- **GitHub Secrets**: Manage secrets for secure CI/CD pipelines.

## 🎯 Project Workflow Summary
- Data Ingestion ➔ Data Validation ➔ Data Transformation
- Model Training ➔ Model Evaluation ➔ Model Deployment
- CI/CD Automation with GitHub Actions, Docker, AWS EC2, and ECR
