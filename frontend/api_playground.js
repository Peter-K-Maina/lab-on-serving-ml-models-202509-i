function getBaseUrl() {
    return document.getElementById("baseUrl").value.trim().replace(/\/$/, "");
}

function renderOutput(elementId, data) {
    const outputElement = document.getElementById(elementId);
    outputElement.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

async function callApi({ endpoint, method = "POST", payload = null, outputId }) {
    const baseUrl = getBaseUrl();
    const url = baseUrl + endpoint;

    const options = {
        method,
        headers: {
            "Content-Type": "application/json"
        }
    };

    if (payload !== null) {
        options.body = JSON.stringify(payload);
    }

    try {
        const response = await fetch(url, options);
        const result = await response.json();

        renderOutput(outputId, {
            status: response.status,
            endpoint,
            request_payload: payload,
            response: result
        });
    } catch (error) {
        renderOutput(outputId, "Network or CORS error: " + error.message);
    }
}

document.getElementById("btnHealth").addEventListener("click", async () => {
    await callApi({
        endpoint: "/api/v1/health",
        method: "GET",
        payload: null,
        outputId: "healthOutput"
    });
});

document.getElementById("classifierForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const model = document.getElementById("classifierModel").value;
    const payload = {
        monthly_fee: Number(document.getElementById("monthly_fee").value),
        customer_age: Number(document.getElementById("customer_age").value),
        support_calls: Number(document.getElementById("support_calls").value)
    };

    await callApi({
        endpoint: `/api/v1/models/${model}/predictions`,
        payload,
        outputId: "classifierOutput"
    });
});

document.getElementById("regressorForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
        PaymentDate: document.getElementById("PaymentDate").value,
        CustomerType: document.getElementById("CustomerType").value,
        BranchSubCounty: document.getElementById("BranchSubCounty").value,
        ProductCategoryName: document.getElementById("ProductCategoryName").value,
        QuantityOrdered: Number(document.getElementById("QuantityOrdered").value)
    };

    await callApi({
        endpoint: "/api/v1/models/decision-tree-regressor/predictions",
        payload,
        outputId: "regressorOutput"
    });
});

document.getElementById("associationForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
        product: document.getElementById("product").value.trim()
    };

    await callApi({
        endpoint: "/api/v1/models/association-rules-recommender/recommendations",
        payload,
        outputId: "associationOutput"
    });
});

document.getElementById("genericForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const endpoint = document.getElementById("genericEndpoint").value.trim();
    const rawPayload = document.getElementById("genericPayload").value;

    let payload;
    try {
        payload = JSON.parse(rawPayload);
    } catch (error) {
        renderOutput("genericOutput", "Invalid JSON payload: " + error.message);
        return;
    }

    await callApi({
        endpoint,
        payload,
        outputId: "genericOutput"
    });
});
