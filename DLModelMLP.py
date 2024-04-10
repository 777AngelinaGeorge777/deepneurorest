#BL.EN.U4CSE21017: Angelina George
#MLP (Multi Layer Perceptron) (A type of Artificial Neural Network)
# Early Stopping at Epoch 46

#We are not doing the continuous learning (train + test) at the end, in case the user inputs wrong data

#plt.show() is a blocking UI. Don't use it, when you are trying to connect it with the front end.

import io, base64, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import requests

if (not os.path.exists("trained_model.h5")):
    #Should happen only once. Not always.
    # Load your dataset (replace 'SaYoPillow.csv' and 'sl' with your actual dataset and target column)
    data = pd.read_csv("SaYoPillow_AC-GAN.csv")
    target_variable = "sl"

    # Encode class labels for multi-class classification
    label_encoder = LabelEncoder()
    data[target_variable] = label_encoder.fit_transform(data[target_variable])
    num_classes = len(label_encoder.classes_)

    # Split the data into features and target
    X = data.drop(target_variable, axis=1)
    y = to_categorical(data[target_variable], num_classes=num_classes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a simple MLP model for classification
    model = Sequential()
    model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Define early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model
    model.save("trained_model.h5")

    # Print performance metrics
    train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train)
    print(f"\nTraining Loss: {train_loss}")
    print(f"Training Accuracy: {train_accuracy}")

    val_loss, val_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"\nValidation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Print training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    print("Model trained and saved successfully!")

#-----------------------------------------------------------

def save_plt(plt):
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    return base64.b64encode(img.read()).decode()

#-----------------------------------------------------------

def testmodel(url, uid):

    # Load the saved model
    model = load_model("trained_model.h5")
    print(url)
    # Load the testing dataset (replace 'testing_data.xlsx' with your actual testing dataset)
    data = io.StringIO(requests.get(url).text)
    # print(data.getvalue())
    # data.seek(0)
    testing_data = pd.read_csv(data)
    

    # Ensure that the testing data has the same features as the training data used for training the model
    # Perform any necessary preprocessing steps (e.g., scaling) on the testing data
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(testing_data)  # Assuming testing_data contains only features, no labels

    # Perform predictions on the testing data
    predictions = model.predict(X_test_scaled)

    # Get the predicted labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Add the predicted labels to the original testing dataset
    testing_data['predicted_labels'] = predicted_labels

    # Save the dataset with predicted labels
    testing_data.to_csv(f"{uid}.csv", index=False)

#-------------------------------------------------

def plotgraphs(uid):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    # Load your dataset (replace 'testing_data_with_labels.xlsx' with your actual dataset)
    data = pd.read_csv(f"{uid}.csv")

    # Assuming your target variable is 'predicted_labels' and features are all columns except 'predicted_labels'
    X = data.drop(columns=['predicted_labels'])
    y = data['predicted_labels']

    # Standardize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a RandomForestClassifier to get feature importance
    rf = RandomForestClassifier()
    rf.fit(X_scaled, y)

    # Get feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Plot stress levels varying (boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='predicted_labels', data=data)
    plt.title('Distribution of Predicted Stress Levels')
    plt.xlabel('Predicted Stress Level')
    plt.ylabel('Count')
    plt.savefig('predicted_stress_levels_boxplot.png')
    box_plot_stress=save_plt(plt)



    # Plot stress levels varying (lineplot)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=data.index, y='predicted_labels')
    plt.title('Stress Levels Variation')
    plt.xlabel('Data Point Index')
    plt.ylabel('Predicted Stress Level')
    plt.savefig('stress_levels_variation_lineplot.png')
    line_plot_stress=save_plt(plt)


    # Plot feature importance in descending order
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title('Feature Importance (Descending Order)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    feature_importance_stress=save_plt(plt)
    return box_plot_stress, line_plot_stress, feature_importance_stress




