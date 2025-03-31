module input_reuse #(
    parameter int KERNELS = 3,
    parameter int ROWS = 3,
    parameter int COLS = 3,
    parameter int DATA_WIDTH = 32,
    parameter int IMAGE_ROWS = 28,
    parameter int IMAGE_COLS = 28
) (
    input logic clk,
    input logic rst,
    input logic start,
    input logic [DATA_WIDTH-1:0] image[IMAGE_ROWS][IMAGE_COLS],
    output logic done
);

    // Kernel memory
    logic signed [DATA_WIDTH-1:0] kernel_mem[KERNELS][ROWS][COLS];
    
    // Sliding window registers
    logic [DATA_WIDTH-1:0] buffer[ROWS][COLS];
    
    // Convolution result (internal register)
    logic signed [DATA_WIDTH-1:0] conv_result[KERNELS][IMAGE_ROWS-ROWS+1][IMAGE_COLS-COLS+1];
    
    // Internal signals
    logic signed [DATA_WIDTH-1:0] partial_sum[KERNELS];
    integer row_idx, col_idx;

    // FSM states
    typedef enum logic [2:0] {IDLE, LOAD_BUFFER, PARTIAL_SUM, STORE_RESULT, DONE} state_t;
    state_t state;

    // Initialize kernel memory
    integer idx;
    initial begin
        logic signed [3:0] flat_mem[KERNELS*ROWS*COLS];
        $readmemh("./src/files/golden_kernel.mem", flat_mem);
        
        idx = 0;
        for (int k = 0; k < KERNELS; k++) begin
            for (int r = 0; r < ROWS; r++) begin
                for (int c = 0; c < COLS; c++) begin
                    kernel_mem[k][r][c] = flat_mem[idx];
                    $display("kernel_mem[%0d][%0d][%0d] = %0d", k, r, c, kernel_mem[k][r][c]);
                    idx++;
                end
            end
        end
    end

    // Combinational logic for convolution
    always_comb begin
        for (integer k = 0; k < KERNELS; k++) begin
            partial_sum[k] = 0;
            for (integer r = 0; r < ROWS; r++) begin
                for (integer c = 0; c < COLS; c++) begin
                    partial_sum[k] += buffer[r][c] * kernel_mem[k][r][c];
                end
            end
        end
    end

    // Sequential logic for FSM
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            row_idx <= 0;
            col_idx <= 0;
            done <= 0;
            
            // Reset conv_result
            for (integer k = 0; k < KERNELS; k++) begin
                for (integer r = 0; r < IMAGE_ROWS - ROWS + 1; r++) begin
                    for (integer c = 0; c < IMAGE_COLS - COLS + 1; c++) begin
                        conv_result[k][r][c] <= 0;
                    end
                end
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= LOAD_BUFFER;
                    end
                end
                LOAD_BUFFER: begin
                    // Load the sliding window
                    for (integer r = 0; r < ROWS; r++) begin
                        for (integer c = 0; c < COLS; c++) begin
                            buffer[r][c] <= image[row_idx + r][col_idx + c];
                        end
                    end
                    state <= PARTIAL_SUM;
                end

                PARTIAL_SUM: begin
                    state <= STORE_RESULT;
                end

                STORE_RESULT: begin
                    // Store the result from combinational logic
                    for (integer k = 0; k < KERNELS; k++) begin
                        conv_result[k][row_idx][col_idx] <= partial_sum[k];
                    end

                    // Update indices
                    if (col_idx < IMAGE_COLS - COLS) begin
                        col_idx <= col_idx + 1;
                        state <= LOAD_BUFFER;
                    end else if (row_idx < IMAGE_ROWS - ROWS) begin
                        col_idx <= 0;
                        row_idx <= row_idx + 1;
                        state <= LOAD_BUFFER;
                    end else begin
                        state <= DONE;
                    end
                end
                DONE: begin
                    done <= 1;
                    state <= IDLE; // Reset to IDLE for next operation
                end
            endcase
        end
    end

endmodule

