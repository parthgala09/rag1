# Filtered Query Examples

These examples target the indexed NSE Clearing Limited Thirtieth Annual Report 2024-25 corpus.

The top-5 result sections should be populated from real Pinecone output after ingestion:

```bash
python3 ingest.py --pdf-dir vvs --manifest vvs/corpus_manifest.csv
```

## 1. Financial Highlights

```bash
python3 query.py --query "What are the financial highlights of the report?" --year 2025 --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 2. Risk Management

```bash
python3 query.py --query "What does the report say about risk management?" --year 2025 --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 3. Audit Committee

```bash
python3 query.py --query "What are the audit committee findings or responsibilities?" --doc-id 30th_annual_report --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 4. Dividend

```bash
python3 query.py --query "What dividend information is included in the annual report?" --year 2025 --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 5. Corporate Governance

```bash
python3 query.py --query "How does the report describe corporate governance?" --doc-id 30th_annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 6. Board Of Directors

```bash
python3 query.py --query "Who are the directors and what board matters are discussed?" --year 2025 --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 7. Business Operations

```bash
python3 query.py --query "What does the report say about business operations and clearing activities?" --doc-id 30th_annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 8. Compliance

```bash
python3 query.py --query "What compliance and regulatory matters are mentioned?" --year 2025 --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 9. Page-Specific Search

```bash
python3 query.py --query "What is introduced at the start of the annual report?" --doc-id 30th_annual_report --page-number 1 --top-k 5
```

Top 5 retrieved chunks: pending live query run.

## 10. Financial Statements

```bash
python3 query.py --query "What do the financial statements say about assets and liabilities?" --year 2025 --doc-type annual_report --top-k 5
```

Top 5 retrieved chunks: pending live query run.
