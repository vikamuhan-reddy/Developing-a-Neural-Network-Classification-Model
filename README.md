# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset using PyTorch.

## THEORY
A neural network classification model is a supervised learning technique used to classify data into different categories or classes. In this experiment, customer data is analyzed to predict the correct customer segment (A, B, C, or D) for new customers based on their attributes.

The dataset contains customer-related features such as gender, marital status, age, profession, work experience, spending score, and family size. Before training the model, preprocessing techniques such as handling missing values, encoding categorical variables, feature scaling, and train-test splitting are applied.

The neural network consists of an input layer, multiple hidden layers, and an output layer. The hidden layers use the ReLU activation function to learn meaningful patterns from the data. Since the problem involves four classes, the output layer contains four neurons representing customer segments A, B, C, and D.

The model is trained using the Cross Entropy Loss function and the Adam optimizer. During training, the model updates weights through backpropagation to minimize classification error. Finally, the trained model is evaluated using accuracy, confusion matrix, and classification report, and it predicts the class of new customer data.

## Neural Network Model
<img width="754" height="546" alt="Screen Shot 2026-05-11 at 13 21 07" src="https://github.com/user-attachments/assets/8dd26860-d840-4dc5-b638-9856d9a7ce1b" />



## DESIGN STEPS

### STEP 1:
Load the dataset and import the required libraries.

### STEP 2:
Preprocess the dataset by removing unnecessary columns, handling missing values, and encoding categorical variables.

### STEP 3:
Split the dataset into training and testing data and normalize the feature values using StandardScaler.

### STEP 4:
Convert the processed data into tensors and create DataLoader objects for batch processing.

### STEP 5:
Build the neural network model, define the loss function and optimizer, and train the model.

### STEP 6:
Evaluate the model using confusion matrix and classification report, and predict the output for a sample input.

## PROGRAM

### Name: Vikamuhan Reddy

### Register Number: 212223240181

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

## Dataset Information

<img width="690" height="137" alt="Screen Shot 2026-05-11 at 13 25 20" src="https://github.com/user-attachments/assets/89527c14-fced-40d5-a0fe-78ce68134cfc" />



## OUTPUT

### Confusion Matrix
<img width="302" height="233" alt="Screen Shot 2026-05-11 at 13 23 07" src="https://github.com/user-attachments/assets/2c7ab7d0-f1dd-4ef6-8b4a-897cfa236105" />


### Classification Report
<img width="373" height="134" alt="Screen Shot 2026-05-11 at 13 23 31" src="https://github.com/user-attachments/assets/76b7a45c-e692-4b5d-9328-68b296fe8064" />



### New Sample Data Prediction
<img width="221" height="32" alt="Screen Shot 2026-05-11 at 13 24 00" src="https://github.com/user-attachments/assets/895aec18-2003-4ec5-ba64-18e043bba173" />

## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch to classify customer segments. The model was evaluated using confusion matrix and classification report and successfully predicted the customer class for a new sample input.
