(() => {
    const API_BASE_URL = "http://localhost:8080";

    const input = document.getElementById("sudoku-image-input");
    const dropzone = document.querySelector(".upload-dropzone");
    const previewImage = document.querySelector(".upload-preview-image");
    const changeImageBtn = document.querySelector(".change-image-btn");
    const processImageBtn = document.querySelector(".process-image-btn");
    const solveBtn = document.querySelector(".solve-btn");
    const ocrStatus = document.querySelector(".ocr-status");
    const gridCells = Array.from(document.querySelectorAll(".sudoku-grid .cell"));
    const numberButtons = Array.from(document.querySelectorAll(".num-btn"));
    const actionItems = Array.from(document.querySelectorAll(".action-item"));

    if (!input || !dropzone || !previewImage || !changeImageBtn || !processImageBtn || !solveBtn || !ocrStatus || gridCells.length !== 81) {
        return;
    }

    let selectedIndex = null;
    const boardHistory = [];

    const setStatus = (message, isError = false) => {
        ocrStatus.textContent = message;
        ocrStatus.style.display = "block";
        ocrStatus.style.color = isError ? "#b91c1c" : "#1e3a8a";
    };

    const showSolveCelebration = () => {
        const existing = document.querySelector(".solve-celebration");
        if (existing) {
            existing.remove();
        }

        const bubble = document.createElement("div");
        bubble.className = "solve-celebration";
        bubble.textContent = "🎉 Congratulations! Sudoku solved!";
        document.body.appendChild(bubble);

        window.setTimeout(() => {
            bubble.classList.add("hide");
            window.setTimeout(() => bubble.remove(), 500);
        }, 3000);
    };

    const snapshotBoard = () => gridCells.map((cell) => cell.textContent || "");

    const restoreBoard = (snapshot) => {
        if (!Array.isArray(snapshot) || snapshot.length !== 81) {
            return;
        }
        for (let i = 0; i < 81; i++) {
            gridCells[i].textContent = snapshot[i] || "";
        }
    };

    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const animateSolveByColumn = async (fromBoard, solvedBoard) => {
        if (!Array.isArray(fromBoard) || !Array.isArray(solvedBoard)) {
            return;
        }

        const animationOrder = [];
        const visited = new Set();
        const addRegion = (rowStart, rowEnd, colStart, colEnd) => {
            for (let row = rowStart; row <= rowEnd; row++) {
                for (let col = colStart; col <= colEnd; col++) {
                    const key = `${row}-${col}`;
                    if (!visited.has(key)) {
                        visited.add(key);
                        animationOrder.push([row, col]);
                    }
                }
            }
        };

        // Alternate single horizontal row, then single vertical column.
        // Example: row0 -> col0 -> row1 -> col1 -> ...
        for (let i = 0; i < 9; i++) {
            addRegion(i, i, 0, 8);
            addRegion(0, 8, i, i);
        }

        for (const [row, col] of animationOrder) {
            const fromValue = Number(fromBoard[row]?.[col] || 0);
            const toValue = Number(solvedBoard[row]?.[col] || 0);

            if (fromValue === 0 && toValue > 0) {
                const cellIndex = row * 9 + col;
                gridCells[cellIndex].textContent = String(toValue);
                gridCells[cellIndex].classList.add("is-selected");
                await wait(45);
                gridCells[cellIndex].classList.remove("is-selected");
            }
        }
    };

    const pushHistory = () => {
        boardHistory.push(snapshotBoard());
        if (boardHistory.length > 100) {
            boardHistory.shift();
        }
    };

    const setSelectedCell = (index) => {
        selectedIndex = index;
        gridCells.forEach((cell, i) => {
            cell.classList.toggle("is-selected", i === index);
        });
    };

    const readBoardFromGrid = () => {
        const board = Array.from({ length: 9 }, () => Array(9).fill(0));
        for (let i = 0; i < 81; i++) {
            const value = Number(gridCells[i].textContent || 0);
            board[Math.floor(i / 9)][i % 9] = Number.isFinite(value) ? value : 0;
        }
        return board;
    };

    const writeCellValue = (index, value) => {
        if (index == null || index < 0 || index >= 81) {
            return;
        }
        gridCells[index].textContent = value === 0 ? "" : String(value);
    };

    const clearStatus = () => {
        ocrStatus.style.display = "none";
        ocrStatus.textContent = "";
    };

    const renderMatrix = (matrix) => {
        if (!Array.isArray(matrix) || matrix.length !== 9) {
            return;
        }

        let index = 0;
        for (let row = 0; row < 9; row++) {
            if (!Array.isArray(matrix[row]) || matrix[row].length !== 9) {
                return;
            }

            for (let col = 0; col < 9; col++) {
                const value = Number(matrix[row][col]) || 0;
                const cell = gridCells[index++];
                cell.textContent = value === 0 ? "" : String(value);
            }
        }
    };

    const clearPreview = () => {
        if (previewImage.src) {
            URL.revokeObjectURL(previewImage.src);
        }
        previewImage.removeAttribute("src");
        dropzone.classList.remove("has-image", "is-processing", "is-dragging");
        clearStatus();
    };

    const setPreviewFromFile = (file) => {
        if (!file) {
            clearPreview();
            return;
        }

        if (!file.type || !file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            input.value = "";
            clearPreview();
            return;
        }

        if (previewImage.src) {
            URL.revokeObjectURL(previewImage.src);
        }

        const objectUrl = URL.createObjectURL(file);
        previewImage.src = objectUrl;
        dropzone.classList.add("has-image");
        dropzone.classList.remove("is-processing");
        setStatus("Image selected. Click Process Image.");
    };

    const processSelectedImage = async () => {
        const file = input.files && input.files[0];
        if (!file) {
            setStatus("Please select an image first.", true);
            return;
        }

        try {
            processImageBtn.disabled = true;
            dropzone.classList.add("is-processing");
            setStatus("Processing image, please wait...");

            const formData = new FormData();
            formData.append("image", file);

            const response = await fetch(`${API_BASE_URL}/solve-image`, {
                method: "POST",
                body: formData
            });

            const payload = await response.json().catch(() => ({}));

            if (!response.ok) {
                const fallbackMatrix = payload.rawDetected || payload.cleanedDetected || payload.detected;
                if (fallbackMatrix) {
                    renderMatrix(fallbackMatrix);
                }
                const message =
                    (payload?.error && typeof payload.error === "object" ? payload.error.message : payload?.error) ||
                    payload?.message ||
                    `Image processing failed (${response.status}).`;
                setStatus(message, true);
                return;
            }

            const matrixToShow = payload.rawDetected || payload.cleanedDetected || payload.detected || payload.solved;
            if (matrixToShow) {
                pushHistory();
                renderMatrix(matrixToShow);
                setStatus(payload.rawDetected ? "Image processed. Showing raw OCR extraction." : "Image processed successfully.");
            } else {
                setStatus("Processing finished, but no Sudoku matrix was returned.", true);
            }
        } catch (error) {
            console.error("Process image error:", error);
            setStatus("Cannot connect to backend at http://localhost:8080. Start the server and try again.", true);
        } finally {
            processImageBtn.disabled = false;
            dropzone.classList.remove("is-processing");
        }
    };

    input.addEventListener("change", (event) => {
        const file = event.target.files && event.target.files[0];
        setPreviewFromFile(file);
    });

    changeImageBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        input.click();
    });

    processImageBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        processSelectedImage();
    });

    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("is-dragging");
    });

    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");

        const droppedFiles = event.dataTransfer?.files;
        const file = droppedFiles && droppedFiles[0];
        if (!file) {
            return;
        }

        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        setPreviewFromFile(file);
    });

    gridCells.forEach((cell, index) => {
        cell.tabIndex = 0;
        cell.addEventListener("click", () => setSelectedCell(index));
        cell.addEventListener("keydown", (event) => {
            if (event.key >= "1" && event.key <= "9") {
                pushHistory();
                writeCellValue(index, Number(event.key));
                return;
            }
            if (event.key === "Backspace" || event.key === "Delete" || event.key === "0") {
                pushHistory();
                writeCellValue(index, 0);
            }
        });
    });

    numberButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (selectedIndex == null) {
                return;
            }
            const value = Number(button.textContent);
            if (value < 1 || value > 9) {
                return;
            }
            pushHistory();
            writeCellValue(selectedIndex, value);
        });
    });

    const getActionButton = (label) => {
        const item = actionItems.find((entry) => {
            const text = entry.querySelector(".action-label")?.textContent?.trim().toLowerCase();
            return text === label;
        });
        return item?.querySelector(".circle-btn") || null;
    };

    const undoButton = getActionButton("undo");
    const eraseButton = getActionButton("erase");
    const clearAllButton = getActionButton("clear all");

    undoButton?.addEventListener("click", () => {
        const previous = boardHistory.pop();
        if (previous) {
            restoreBoard(previous);
        }
    });

    eraseButton?.addEventListener("click", () => {
        if (selectedIndex == null) {
            return;
        }
        pushHistory();
        writeCellValue(selectedIndex, 0);
    });

    clearAllButton?.addEventListener("click", () => {
        pushHistory();
        gridCells.forEach((cell) => {
            cell.textContent = "";
        });
    });

    solveBtn.addEventListener("click", async () => {
        const board = readBoardFromGrid();

        try {
            solveBtn.disabled = true;
            setStatus("Solving manual puzzle...");

            const response = await fetch(`${API_BASE_URL}/solve-manual`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ board })
            });

            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                const message =
                    (payload?.error && typeof payload.error === "object" ? payload.error.message : payload?.error) ||
                    payload?.message ||
                    `Solve failed (${response.status}).`;
                setStatus(message, true);
                return;
            }

            if (payload?.solved) {
                pushHistory();
                await animateSolveByColumn(board, payload.solved);
                renderMatrix(payload.solved);
                setStatus("Sudoku solved successfully.");
                showSolveCelebration();
            } else {
                setStatus("Solver returned no solved board.", true);
            }
        } catch (error) {
            console.error("Manual solve error:", error);
            setStatus("Cannot connect to backend solver.", true);
        } finally {
            solveBtn.disabled = false;
        }
    });
})();