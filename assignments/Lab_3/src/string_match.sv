module string_match (
    input logic clk,
    input logic rst,
    input logic [95:0] search_string,  // 96-bit input string
    output logic found,                // 1 if found, 0 if not found
    output logic signed [31:0] position // Start position of match (-1 if not found)
);

    localparam MEMORY_SIZE = 2028;  // Example memory size (32-bit words)
    logic [31:0] memory [0:MEMORY_SIZE-1]; // Memory array (32-bit words)

    initial begin
        // Load data from file into memory
        $readmemh("golden_conv.mem", memory);
    end
    
    // TO DO COMPLETE
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            found <= 0;
            position <= -1;
        end else begin
            found <= 0;
            position <= -1;
            for (int i = 0; i < MEMORY_SIZE - 2; i++) begin
                if (
                    memory[i]   == search_string[31:0] &&
                    memory[i+1] == search_string[63:32] &&
                    memory[i+2] == search_string[95:64]
                ) begin
                    found <= 1;
                    position <= i;
                end
            end
        end
    end
endmodule