# Churn-prediction
Prédiction de client frauduleux
Le modèle est crée dans le fichier churn.py
Les fichiers train.csv et test.csv contiennet des données sur différents clients et sont issues de la base de données Kaggle (Customers Churn prediction 2020) (https://www.kaggle.com/c/customer-churn-prediction-2020).

Le Modèle testé ici, est un réseau de neuronnes à une couche cachée. La bibliothèque utilisée est Tensorflow (Keras). Pour évaluer le modèle, nous avons utilisé les mesures suivantes : Accuracy, Precision, Recall et F1-score.

              precision    recall  f1-score   support

           0       0.90      0.99      0.94       738
           1       0.74      0.28      0.40       112

    accuracy                           0.89       850
   macro avg       0.82      0.63      0.67       850
weighted avg       0.88      0.89      0.87       850

On conclut donc, que le modèle est efficace pour reconnaître les clients non frauduleux (f1-score de 0.94), cependant le modèle ne reconnaît que 40% des clients frauduleux, ce qui semble assez faibles, en revanche parmi les client que le modèle considère comme frauduleux, 74% le sont vraiment. Ainsi le modèle semble plutôt fiable lorsqu'il classifie un clien comme frauduleux.
