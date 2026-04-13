# modeling/trainer.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class ModelTrainer:
    def __init__(self, model):
        """
        Trainer receives a model instance.
        """
        self.model = model

    def train(self, X_train, y_train):
        """
        Train the model.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        """
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        print("\n[Model Evaluation]")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"ROC-AUC:   {roc_auc:.3f}")


