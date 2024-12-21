document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById("result").textContent = `Prediction: ${data.prediction}`;
        } else {
            const errorData = await response.json();
            alert(`Error: ${errorData.error}`);
        }
    } catch (err) {
        alert("An error occurred while processing your request.");
    }
});
