from metaflow import FlowSpec, step, Parameter

class ClassifierTrainFlow(FlowSpec):
    
    n_neighbors = Parameter('n_neighbors', default=5, type=int)
    weights = Parameter('weights', default='uniform')

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X,y, test_size=0.2)
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
            self.train_data, self.train_labels, test_size=0.25)
        print("Data loaded successfully")
        
        # Hyperparameter combinations
        self.param_combos = []
        # For each neighbor
        for n in range(1, 10):
            # For each weight
            for w in ['uniform', 'distance']:
                self.param_combos.append((n, w))
                
        self.next(self.train_knn, foreach='param_combos')

    @step
    def train_knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        
        #Train model
        n, w = self.input
        self.model = KNeighborsClassifier(n_neighbors=n, weights=w)
        self.model.fit(self.train_data, self.train_labels)
        
        # Save for next step
        self.n_neighbors_value = n
        self.weights_value = w
        
        # Evaluate model
        self.score_value = self.model.score(self.val_data, self.val_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        import mlflow
        # mlflow.set_tracking_uri('https://mlflow-test-run-275570243848.us-west2.run.app')
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('lab6-experiment')

        # Create list of models and info
        models_with_scores = []
        for inp in inputs:
            model_info = (inp.model, 
                          inp.score_value, 
                          {'n_neighbors': inp.n_neighbors_value, 
                           'weights': inp.weights_value})
            models_with_scores.append(model_info)
        
        # Sort and get best model
        self.results = sorted(models_with_scores, key=lambda x: -x[1])
        self.model = self.results[0][0]
        best_params = self.results[0][2]
        
        # Log the model and the parameters
        with mlflow.start_run():
            # Log model, parameters, and score
            mlflow.log_param("n_neighbors", best_params["n_neighbors"])
            mlflow.log_param("weights", best_params["weights"])
            mlflow.log_metric("accuracy", self.results[0][1])
            mlflow.sklearn.log_model(self.model, 
                                     artifact_path='metaflow_train', 
                                     registered_model_name="metaflow-wine-model")
        mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        best_model, best_score, best_params = self.results[0]
        print('Best model:')
        print('Model:', best_model)
        print('Score:', best_score)
        print('Neighbors:', best_params['n_neighbors'])
        print('Weights:', best_params['weights'])


if __name__=='__main__':
    ClassifierTrainFlow()