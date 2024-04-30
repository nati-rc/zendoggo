document.getElementById('audioForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    document.querySelector('.spinner-border').style.display = 'block'; // Show spinner
    fetch('/analyze', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        document.querySelector('.spinner-border').style.display = 'none'; // Hide spinner
        const results = data.analysis_results;
        const geminiResults = data.gemini_response;
        const totalAudioLength = 'Total Audio Length: ' + results.total_audio_length_seconds + ' seconds';
        let tableHtml = '<h3>Audio Breakdown</h3><div class="scrollable-table"><table class="table"><thead><tr><th>Start Time</th><th>End Time</th><th>Category</th></tr></thead><tbody>';

        for (const segment of results.segments) {
            tableHtml += `<tr><td>${segment.start_time.toFixed(2)}s</td><td>${segment.end_time.toFixed(2)}s</td><td>${segment.category}</td></tr>`;
        }

        tableHtml += '</tbody></table></div>';
        let geminiHtml = `<div class='gemini-results'><h3>Analysis</h3><p><strong>Summary: </strong>${geminiResults["Percentage Distribution"]}</p><p><strong>Insights: </strong> ${geminiResults["Summary"]}</p><h4>Suggestions:</h4><ul>`;

        if (Array.isArray(geminiResults["Suggestions"])) {
            geminiResults["Suggestions"].forEach(suggestion => {
                geminiHtml += `<li>${suggestion}</li>`;
            });
        } else {
            Object.keys(geminiResults["Suggestions"]).forEach(key => {
                geminiHtml += `<li>${geminiResults["Suggestions"][key]}</li>`;
            });
        }

        geminiHtml += '</ul></div>';

        document.getElementById('results').innerHTML = `<h3>${totalAudioLength}</h3>` + geminiHtml + tableHtml;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = '<div class="alert alert-danger">Error: ' + error.toString() + '</div>';
    });
});

