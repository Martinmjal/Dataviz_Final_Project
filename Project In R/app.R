library(shiny)
library(ggplot2)
library(DT)
library(plotly)
library(dplyr)
library(reshape2)
library(xgboost)
library(shinythemes)
library(randomForest)
library(pROC)

# Read the data for eda
df <- read.csv("final_dataframe_churn_prediction.csv")
df$is_churned_label <- as.factor(ifelse(df$is_churned == 0, 'not churned', 'churned'))

numerical_cols <- setdiff(names(df), c('warehouse_Bogota', 'sub_categoria_Ambientadores', 'sub_categoria_Complementos.y.vitaminas', 'is_churned', 'is_churned_label'))
categorical_cols <- c("warehouse_Bogota","sub_categoria_Ambientadores","sub_categoria_Complementos.y.vitaminas")

# Read the train test data

test_set <- read.csv("test_set.csv")
test_set$warehouse_Bogota <- factor(test_set$warehouse_Bogota)
test_set$sub_categoria_Ambientadores <- factor(test_set$sub_categoria_Ambientadores)
test_set$sub_categoria_Complementos.y.vitaminas <- factor(test_set$sub_categoria_Complementos.y.vitaminas)
test_set$is_churned <- factor(test_set$is_churned)

train_set <- read.csv("train_set.csv")
train_set$warehouse_Bogota <- factor(train_set$warehouse_Bogota)
train_set$sub_categoria_Ambientadores <- factor(train_set$sub_categoria_Ambientadores)
train_set$sub_categoria_Complementos.y.vitaminas <- factor(train_set$sub_categoria_Complementos.y.vitaminas)
train_set$is_churned <- factor(train_set$is_churned)

# Custom CSS for styling
customCSS <- "
body {
  background-color: #f4f4f4;
  font-family: 'Arial', sans-serif;
}
.navbar {
  background-color: #007bff;
}
.navbar .navbar-brand {
  color: #fff;
}
.navbar .navbar-nav>li>a {
  color: #fff;
}
.titlePanel {
  background-color: #007bff;
  color: #fff;
  padding: 15px;
  border-radius: 5px;
  text-align: center;
}
.well {
  background-color: #e9ecef;
  padding: 15px;
  margin-bottom: 20px;
  border-radius: 5px;
}
.selectInput, .numericInput {
  margin-bottom: 20px;
}
.container-fluid {
  padding: 20px;
}
"

# UI
ui <- fluidPage(
  tags$head(tags$style(HTML(customCSS))),
  titlePanel(
    "Customer Classification Model: Churn Prediction", 
    windowTitle = "Churn Prediction Dashboard"
  ),
  
  # Tabs
  navbarPage(
    "",
    tabPanel(
      "EDA",
      div(class = "well",
          selectInput(
            "numericalVar", 
            "Select Numerical Variable:", 
            choices = names(df[, numerical_cols]), 
            selected = "antiquity_days"
          )
      ),
      fluidRow(
        column(6, plotlyOutput("boxPlot", height = "400px"), align = "center"),
        column(6, plotlyOutput("histogram", height = "400px"), align = "center")
      ),
      div(class = "well",
          selectInput(
            "categoricalVar", 
            "Select Categorical Variable", 
            choices = names(df[, categorical_cols]),
            selected = "warehouse_Bogota"
          )
      ),
      fluidRow(
        column(6, plotlyOutput("stacked_barchart", height = "400px"), align = "center"),
        column(6, plotlyOutput("barchart", height = "400px"), align = "center")
      )
    ),
    tabPanel(
      "Classifiers",
      div(class = "well",
          selectInput(
            "modelChoice", 
            "Select Model:", 
            choices = c(
              "Random Forest" = "random_forest_model.rds", 
              "Logistic Regression" = "logistic_regression.rds", 
              "XGBoost" = "xgboost_model.rds"
            ),
            selected = "random_forest_model.rds"
          )
      ),
      fluidRow(
        column(6, plotOutput("confMatrixPlot", height = "400px")),
        column(6, plotOutput("rocCurvePlot", height = "400px"))
      ),
      dataTableOutput("modelMetrics")
    )
  )
)

# Server
server <- function(input, output) {
  
  # Custom Theme for ggplot2
  customTheme <- function() {
    theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
        axis.title = element_text(size = 8),
        axis.text = element_text(size = 8),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "#f4f4f4"),
        plot.background = element_rect(fill = "#f4f4f4")
      )
  }
  
  # Reactive expression for data processing at stacked barplot
  processedData <- reactive({
    df_percent <- df %>%
      group_by_at(input$categoricalVar) %>%
      count(is_churned_label) %>%
      group_by_at(input$categoricalVar) %>%
      mutate(perc = n / sum(n) * 100)
    return(df_percent)
  })
  
  # Boxplot Output with custom styling
  output$boxPlot <- renderPlotly({
    ggplot(df, aes_string(x = "is_churned_label", y = paste(input$numericalVar))) +
      geom_boxplot(fill = "#007bff", color = "#333333") +
      coord_flip() +
      labs(title = paste("Boxplot of", input$numericalVar, "by is_churned"),
           y = input$numericalVar,
           x = "is_churned") +
      customTheme()
  })
  
  # Histogram Output with custom styling
  output$histogram <- renderPlotly({
    ggplot(df, aes_string(x = paste(input$numericalVar))) +
      geom_histogram(fill = "#007bff", bins = 30) +
      labs(title = paste("Histogram of", input$numericalVar)) +
      customTheme()
  })
  
  # Stacked Bar Chart with custom styling
  output$stacked_barchart <- renderPlotly({
    df_percent <- processedData()
    ggplot(df_percent, aes_string(x = input$categoricalVar, y = "perc", fill = "is_churned_label")) +
      geom_bar(stat = "identity", position = "fill") +
      scale_fill_manual(values = c("churned" = "#007bff", "not churned" = "#ff7f00")) +
      labs(title = paste("Percentage of is_churned by", input$categoricalVar),
           x = input$categoricalVar,
           y = "Percentage (%)") +
      customTheme()
  })

  # Barplot
  output$barchart <- renderPlotly({
    ggplot(data = df, aes(x = is_churned_label)) +
      geom_bar(fill = "#007bff") +
      labs(title = "Count of churned and non churned customers",
           x = "",
           y = "Count") +
      customTheme()
  })
  
  # Reactive expression for confusion matrix
  output$confMatrixPlot <- renderPlot({
    req(input$modelChoice) # Ensure modelChoice is selected
    
    if(input$modelChoice == "random_forest_model.rds") {
      model <- readRDS(input$modelChoice)
      predictions_prob <- predict(model, newdata = test_set, type = 'prob')
      prob_positive_class <- predictions_prob[, "1"]
      predicted_class <- ifelse(prob_positive_class > 0.5, 1, 0)
      confusion_matrix <- table(test_set$is_churned, predicted_class)
      confusion_df <- as.data.frame(melt(confusion_matrix))
      colnames(confusion_df) <- c("Reference", "Prediction", "Count")
      
      conf_matr_chart <- ggplot(confusion_df, aes(x = Reference, y = Prediction, fill = Count)) +
        geom_tile() +
        geom_text(aes(label = Count), vjust = 1) +
        scale_fill_gradient(low = "white", high = "blue") +
        theme_minimal() +
        labs(title = "Confusion Matrix", x = "Actual Class", y = "Predicted Class", fill = "Count") +
        customTheme()
      return(conf_matr_chart)
    }
    
    if(input$modelChoice == "logistic_regression.rds") {
      model <- readRDS(input$modelChoice)
      predictions <- predict(model, newdata = test_set, type = "response")
      predicted_class <- ifelse(predictions > 0.5, 1, 0)
      confusion_matrix <- table(test_set$is_churned, predicted_class)
      confusion_df <- as.data.frame(melt(confusion_matrix))
      colnames(confusion_df) <- c("Reference", "Prediction", "Count")
      
      conf_matr_chart <- ggplot(confusion_df, aes(x = Reference, y = Prediction, fill = Count)) +
        geom_tile() +
        geom_text(aes(label = Count), vjust = 1) +
        scale_fill_gradient(low = "white", high = "blue") +
        theme_minimal() +
        labs(title = "Confusion Matrix", x = "Actual Class", y = "Predicted Class", fill = "Count") +
        customTheme()
      return(conf_matr_chart)
    }
    
    if(input$modelChoice == "xgboost_model.rds") {
      model <- readRDS(input$modelChoice)
      
      num_conversion <- function(df) {
        for (col in names(df)) {
          if (is.factor(df[[col]])) {
            df[[col]] <- as.numeric(as.character(df[[col]]))
          }
        }
        return(df)
      }
      
      train_set_num <- num_conversion(train_set)
      test_set_num <- num_conversion(test_set)
      
      # convert to matrix
      train_data <- as.matrix(train_set_num[, -which(names(train_set_num) == "is_churned")])
      train_label <- train_set_num$is_churned
      
      test_data <- as.matrix(test_set_num[, -which(names(test_set_num) == "is_churned")])
      test_label <- test_set_num$is_churned
      
      dtrain <- xgb.DMatrix(data = train_data, label = train_label)
      dtest <- xgb.DMatrix(data = test_data, label = test_label)
      
      predictions <- predict(model, dtest)
      predicted_class <- ifelse(predictions > 0.5, 1, 0)
      
      confusion_matrix <- table(test_label, predicted_class)
      confusion_df <- as.data.frame(melt(confusion_matrix))
      colnames(confusion_df) <- c("Reference", "Prediction", "Count")
      
      conf_matr_chart <- ggplot(confusion_df, aes(x = Reference, y = Prediction, fill = Count)) +
        geom_tile() +
        geom_text(aes(label = Count), vjust = 1) +
        scale_fill_gradient(low = "white", high = "blue") +
        theme_minimal() +
        labs(title = "Confusion Matrix", x = "Actual Class", y = "Predicted Class", fill = "Count") +
        customTheme()
      return(conf_matr_chart)
    }
    
  })
  
  # Reactive expression for ROC curve
  output$rocCurvePlot <- renderPlot({
    req(input$modelChoice) # Ensure modelChoice is selected
    
    if(input$modelChoice == "random_forest_model.rds") {
      model <- readRDS(input$modelChoice)
      predictions_prob <- predict(model, newdata = test_set, type = 'prob')
      prob_positive_class <- predictions_prob[, "1"]
      
      # Create ROC curve data
      roc_curve <- roc(response = as.numeric(as.character(test_set$is_churned)), predictor = prob_positive_class)
      roc_data <- data.frame(
        Sensitivity = roc_curve$sensitivities,
        Specificity = roc_curve$specificities,
        Thresholds = roc_curve$thresholds
      )
      
      # Plot the ROC curve using ggplot2
      roc_curve_plot <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity)) +
        geom_line() +
        geom_abline(linetype = "dashed") +
        xlim(0, 1) + ylim(0, 1) +  # Set x and y axis limits
        labs(title = "ROC Curve for Random Forest", x = "False Positive Rate", y = "True Positive Rate") +
        customTheme()
      return(roc_curve_plot)
    }
    
    if(input$modelChoice == "logistic_regression.rds") {
      model <- readRDS(input$modelChoice)
      
      predictions <- predict(model, newdata = test_set, type = "response")
      predicted_class <- ifelse(predictions > 0.5, 1, 0)
      
      # Create ROC curve data
      roc_curve <- roc(response = as.numeric(as.character(test_set$is_churned)), predictor = predictions)
      roc_data <- data.frame(
        Sensitivity = roc_curve$sensitivities,
        Specificity = roc_curve$specificities,
        Thresholds = roc_curve$thresholds
      )
      
      # Plot the ROC curve using ggplot2
      roc_curve_plot <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity)) +
        geom_line() +
        geom_abline(linetype = "dashed") +
        xlim(0, 1) + ylim(0, 1) +  # Set x and y axis limits
        labs(title = "ROC Curve for Random Forest", x = "False Positive Rate", y = "True Positive Rate") +
        customTheme()
      return(roc_curve_plot)
    }
    
    if(input$modelChoice == "xgboost_model.rds") {
      model <- readRDS(input$modelChoice)
      
      num_conversion <- function(df) {
        for (col in names(df)) {
          if (is.factor(df[[col]])) {
            df[[col]] <- as.numeric(as.character(df[[col]]))
          }
        }
        return(df)
      }
      
      train_set_num <- num_conversion(train_set)
      test_set_num <- num_conversion(test_set)
      
      # convert to matrix
      train_data <- as.matrix(train_set_num[, -which(names(train_set_num) == "is_churned")])
      train_label <- train_set_num$is_churned
      
      test_data <- as.matrix(test_set_num[, -which(names(test_set_num) == "is_churned")])
      test_label <- test_set_num$is_churned
      
      dtrain <- xgb.DMatrix(data = train_data, label = train_label)
      dtest <- xgb.DMatrix(data = test_data, label = test_label)
      
      predictions <- predict(model, dtest)
      predicted_class <- ifelse(predictions > 0.5, 1, 0)
      
      # Create ROC curve data
      roc_curve <- roc(response = as.numeric(as.character(test_set$is_churned)), predictor = predictions)
      roc_data <- data.frame(
        Sensitivity = roc_curve$sensitivities,
        Specificity = roc_curve$specificities,
        Thresholds = roc_curve$thresholds
      )
      
      # Plot the ROC curve using ggplot2
      roc_curve_plot <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity)) +
        geom_line() +
        geom_abline(linetype = "dashed") +
        xlim(0, 1) + ylim(0, 1) +  # Set x and y axis limits
        labs(title = "ROC Curve for Random Forest", x = "False Positive Rate", y = "True Positive Rate") +
        customTheme()
      return(roc_curve_plot)
    }
  })
  
  
  # Reactive expression for model metrics
  output$modelMetrics <- renderDataTable({
    req(input$modelChoice) # Ensure modelChoice is selected
    
    if(input$modelChoice == "random_forest_model.rds") {
      model <- readRDS(input$modelChoice)
      predictions_prob <- predict(model, newdata = test_set, type = 'prob')
      prob_positive_class <- predictions_prob[, "1"]
      predicted_class <- ifelse(prob_positive_class > 0.5, 1, 0)
      
      confusion_matrix <- table(test_set$is_churned, predicted_class)
      
      precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
      recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
      accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
      f1_score <- 2 * (precision * recall) / (precision + recall)

      roc_curve <- roc(response = test_set$is_churned, predictor = prob_positive_class)
      auc_value <- auc(roc_curve)
      
      # Create a data frame for the metrics
      metrics_df <- data.frame(
        Metric = c("Precision", "Recall", "Accuracy", "F1 Score", "AUC"),
        Value = c(precision, recall, accuracy, f1_score, auc_value)
      )
      
      return(metrics_df)
    }
    
    if(input$modelChoice == "logistic_regression.rds") {
      model <- readRDS(input$modelChoice)
      predictions <- predict(model, newdata = test_set, type = "response")
      predicted_class <- ifelse(predictions > 0.5, 1, 0)
      confusion_matrix <- table(test_set$is_churned, predicted_class)
      
      precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
      recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
      accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
      f1_score <- 2 * (precision * recall) / (precision + recall)
      
      roc_curve <- roc(response = test_set$is_churned, predictor = predictions)
      auc_value <- auc(roc_curve)
      
      # Create a data frame for the metrics
      metrics_df <- data.frame(
        Metric = c("Precision", "Recall", "Accuracy", "F1 Score", "AUC"),
        Value = c(precision, recall, accuracy, f1_score, auc_value)
      )
      
      return(metrics_df)
    }
    
    if(input$modelChoice == "xgboost_model.rds") {
      model <- readRDS(input$modelChoice)
      
      num_conversion <- function(df) {
        for (col in names(df)) {
          if (is.factor(df[[col]])) {
            df[[col]] <- as.numeric(as.character(df[[col]]))
          }
        }
        return(df)
      }
      
      train_set_num <- num_conversion(train_set)
      test_set_num <- num_conversion(test_set)
      
      # convert to matrix
      train_data <- as.matrix(train_set_num[, -which(names(train_set_num) == "is_churned")])
      train_label <- train_set_num$is_churned
      
      test_data <- as.matrix(test_set_num[, -which(names(test_set_num) == "is_churned")])
      test_label <- test_set_num$is_churned
      
      dtrain <- xgb.DMatrix(data = train_data, label = train_label)
      dtest <- xgb.DMatrix(data = test_data, label = test_label)
      
      predictions <- predict(model, dtest)
      predicted_class <- ifelse(predictions > 0.5, 1, 0)
      
      confusion_matrix <- table(test_label, predicted_class)
      
      precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
      recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
      accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
      f1_score <- 2 * (precision * recall) / (precision + recall)
      
      roc_curve <- roc(response = test_set$is_churned, predictor = predictions)
      auc_value <- auc(roc_curve)
      
      # Create a data frame for the metrics
      metrics_df <- data.frame(
        Metric = c("Precision", "Recall", "Accuracy", "F1 Score", "AUC"),
        Value = c(precision, recall, accuracy, f1_score, auc_value)
      )
      
      return(metrics_df)
    }
    
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)