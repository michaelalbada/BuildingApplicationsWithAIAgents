# Agent Evaluation Framework

This document explains how to run batch evaluations for our AI agents across different domains.

## Overview

The evaluation framework supports multiple agent types with standardized metrics:
- **Tool Recall**: How well the agent calls the expected tools
- **Tool Precision**: Accuracy of tool calls (no unnecessary tools)
- **Parameter Accuracy**: Correctness of tool parameters
- **Phrase Recall**: Presence of expected phrases in responses
- **Task Success**: Overall task completion score

## Available Evaluation Sets

### 1. E-commerce Customer Support
**Dataset**: `src/common/evaluation/scenarios/ecommerce_customer_support_evaluation_set.json`
**Agent**: `src/frameworks/langgraph_agents/ecommerce_customer_support/customer_support_agent.py`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/ecommerce_customer_support_evaluation_set.json \
  --graph_py src/frameworks/langgraph_agents/ecommerce_customer_support/customer_support_agent.py
```

**Scenarios**: Order refunds, cancellations, address modifications

---

### 2. Financial Services Account Management
**Dataset**: `src/common/evaluation/scenarios/financial_services_account_management.jsonl`
**Agent**: `src/frameworks/langgraph_agents/financial_services/financial_services_agent.py`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/financial_services_account_management.jsonl \
  --graph_py src/frameworks/langgraph_agents/financial_services/financial_services_agent.py
```

**Scenarios**: Fraud detection, account freezing, loan applications, dispute resolution, investment management

---

### 3. Healthcare Patient Intake & Triage
**Dataset**: `src/common/evaluation/scenarios/healthcare_patient_intake_and_triage.jsonl`
**Agent**: `src/frameworks/langgraph_agents/healthcare/healthcare_patient_intake_agent.py`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/healthcare_patient_intake_and_triage.jsonl \
  --graph_py src/frameworks/langgraph_agents/healthcare/healthcare_patient_intake_agent.py
```

**Scenarios**: Patient registration, symptom assessment, appointment scheduling, insurance verification

---

### 4. Security Operations Center (SOC) Analyst
**Dataset**: `src/common/evaluation/scenarios/security_operations_center_analyst.jsonl`
**Agent**: `[Create your SOC agent]`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/security_operations_center_analyst.jsonl \
  --graph_py path/to/your/soc_agent.py
```

**Scenarios**: Threat investigation, log analysis, incident triage, host isolation

---

### 5. IT Help Desk & System Administration
**Dataset**: `src/common/evaluation/scenarios/it_help_desk_system_administration.jsonl`
**Agent**: `[Create your IT help desk agent]`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/it_help_desk_system_administration.jsonl \
  --graph_py path/to/your/it_helpdesk_agent.py
```

**Scenarios**: User access provisioning, password resets, system troubleshooting, software installations

---

### 6. Legal Document Review & Case Management
**Dataset**: `src/common/evaluation/scenarios/legal_document_review_case_management.jsonl`
**Agent**: `[Create your legal agent]`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/legal_document_review_case_management.jsonl \
  --graph_py path/to/your/legal_agent.py
```

**Scenarios**: Contract review, case research, client intake, compliance monitoring

---

### 7. Supply Chain & Logistics Management
**Dataset**: `src/common/evaluation/scenarios/supply_chain_logistics_management.jsonl`
**Agent**: `[Create your supply chain agent]`

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/supply_chain_logistics_management.jsonl \
  --graph_py path/to/your/supply_chain_agent.py
```

**Scenarios**: Inventory management, shipment tracking, vendor relations, warehouse operations

## Advanced Usage

### Custom Metric Weights
You can adjust the importance of different metrics:

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset path/to/dataset.jsonl \
  --graph_py path/to/agent.py \
  --weights task_success=2.0 tool_recall=1.5 param_accuracy=1.0
```

### Verbose Output
Enable detailed debugging information:

```bash
python -m src.common.evaluation.batch_evaluation \
  --dataset path/to/dataset.jsonl \
  --graph_py path/to/agent.py \
  --verbose true
```

## Understanding Results

### Sample Output 