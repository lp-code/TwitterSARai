provider "azurerm" {
  version = "~> 1.41.0"
}

locals {
  storage_account_name = "sa${var.solution_name}"
  service_plan_name    = "sp${var.solution_name}"
  app_insights_name    = "appinsights${var.solution_name}"
  function_app_name    = "fa${var.solution_name}"
  tags                 = { project = "tftest", account = "free" }
}

# Create a resource group
resource "azurerm_resource_group" "rgTSai" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_storage_account" "sa" {
  name                     = local.storage_account_name
  resource_group_name      = azurerm_resource_group.rgTSai.name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_app_service_plan" "asp" {
  name                = local.service_plan_name
  location            = var.location
  resource_group_name = azurerm_resource_group.rgTSai.name
  kind                = "Linux"
  reserved            = true

  sku {
    tier = "Dynamic"
    size = "Y1"
  }
}

resource "azurerm_application_insights" "app_insights" {
  name                = local.app_insights_name
  location            = var.location
  resource_group_name = azurerm_resource_group.rgTSai.name
  application_type    = "web"
}

resource "azurerm_function_app" "example" {
  name                      = local.function_app_name
  resource_group_name       = azurerm_resource_group.rgTSai.name
  location                  = var.location
  app_service_plan_id       = azurerm_app_service_plan.asp.id
  storage_connection_string = azurerm_storage_account.sa.primary_connection_string
  https_only                = true
  enabled = true
  app_settings = {
    FUNCTIONS_WORKER_RUNTIME       = "python"
    WEBSITE_NODE_DEFAULT_VERSION   = "10.14.1"
    APPINSIGHTS_INSTRUMENTATIONKEY = azurerm_application_insights.app_insights.instrumentation_key
  }
  site_config {
      use_32_bit_worker_process = true
  }
  tags = {}
}
