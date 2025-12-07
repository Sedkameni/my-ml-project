# Model Versioning and Rollback Strategy

This document outlines the strategy for managing model versions and the procedure for safely rolling back a deployed API image.

## 1. Automated Versioning and Traceability

In this Continuous Integration/Continuous Deployment (CI/CD) setup, every deployed model version is tied directly to the Git commit that produced it, ensuring full **traceability** and **reproducibility**.

* **Source of Truth (Model Version):** The specific model version (e.g., `1.0`) is exposed by the Flask API via the `/` and `/health` endpoints.
* **Docker Image Tagging:** Our CI/CD pipeline (GitHub Actions) automatically builds and pushes a new Docker image to Docker Hub upon every successful push to the `main` branch. This image is uniquely tagged using the **short Git commit hash** (e.g., `48016cd`).
* **The Tag:** The resulting image tag (`sedkamen/sentiment-api:48016cd`) serves as the definitive version identifier, referencing the exact code, model files, and dependencies used to build a specific API version.

## 2. Rollback Procedure

To roll back the deployed API to a previous, stable version, you simply instruct your deployment environment to pull and run the Docker image associated with that older version's successful commit hash. 

**Procedure:**

1.  **Identify Target Version:** Locate the short commit hash of the last known stable deployment by checking the successful runs in the **GitHub Actions** tab.
2.  **Stop Current Deployment:** Stop the currently running Docker container or deployment service.
3.  **Redeploy Old Image:** Update the deployment command or configuration (e.g., in your Docker Compose or Kubernetes manifest) to use the previous, stable tag.

    ```bash
    # Example: Rolling back to an older commit hash, 'a5c498c'
    docker pull sedkamen/sentiment-api:a5c498c 
    docker run -d -p 80:5000 sedkamen/sentiment-api:a5c498c
    ```

4.  **Verify:** After redeployment, confirm the rollback was successful by querying the API's `/health` or `/` endpoint to ensure the older `model_version` is reported.