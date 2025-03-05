num_boost_rounds = 10
base_estimators = [
    DecisionTreeRegressor(max_depth=2),
    ElasticNet(),
    SVR(kernel='rbf'),
    RandomForestRegressor(max_depth=15, n_estimators=100),
    KNeighborsRegressor(n_neighbors=9)
]

for base_estimator in base_estimators:
    # Create AdaBoostRegressor with the specified base estimator
    ada_boost_model = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=10, random_state=42)
    
    # Fit the model
    ada_boost_model.fit(x_train_scaled, y_train)

for boost_round in range(num_boost_rounds):
    # Assign weights to samples based on the current neural network's predictions
    sample_weights = np.abs(ada_boost_model.estimators_[boost_round].predict(x_train_scaled)-y_train) +1
    # Train the neural network with the weighted samples
    inverted_adaboost.fit(x_train_scaled, y_train, sample_weight=sample_weights, epochs=50, verbose=0)
    # Evaluate the neural network on the test set
    test_loss = inverted_adaboost.evaluate(x_val_scaled, y_test, verbose=0)
    print(f"Boosting Round {boost_round + 1}, Test Loss: {test_loss}")

# Evaluate the neural network on the test set
test_loss = inverted_adaboost.evaluate(x_val_scaled, y_test, verbose=0)
print(f"Boosting Round {boost_round + 1}, VAL Loss: {test_loss}")

final_test_loss = inverted_adaboost.evaluate(x_val_scaled, y_test)
print(f"Final Test Loss: {final_test_loss}")
ada_pred=inverted_adaboost.predict(x_val_scaled)

test_losses.append(test_loss)
min_test_loss = float('inf')

avg_test_loss = np.mean(test_losses)

    # Update the best estimator if the current one has a lower test loss
if avg_test_loss < min_test_loss:
    min_test_loss = avg_test_loss
    best_estimator = base_estimator

# Final evaluation on the test set using the best estimator
final_test_loss = inverted_adaboost.evaluate(x_val_scaled, y_test)
print(f"Final Test Loss using the Best Estimator: {final_test_loss}")
print(f"Best Estimator: {best_estimator.__class__.__name__}")

final_test_loss = mean_squared_error(y_test, ada_pred)
rmse = np.sqrt(final_test_loss)

print(f"Final RMSE using the Best Estimator: {rmse}")
print(f"Best Estimator: {best_estimator.__class__.__name__}")




