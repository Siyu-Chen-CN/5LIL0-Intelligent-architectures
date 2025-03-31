`timescale 1ns/1ps

module tb_convolution_reuse;

    // Parameters
    parameter int KERNELS = 3;          // Number of kernels
    parameter int ROWS    = 3;          // Kernel rows
    parameter int COLS    = 3;          // Kernel columns
    parameter int DATA_WIDTH = 32;      // Bit-width of data
    parameter int IMAGE_ROWS = 28;      // Input image rows
    parameter int IMAGE_COLS = 28;      // Input image columns

    // Inputs and Outputs
    logic clk;
    logic rst;
    logic start;
    logic [DATA_WIDTH-1:0] image[IMAGE_ROWS][IMAGE_COLS];

    // Instantiate the module
    input_reuse #(
        .KERNELS(KERNELS),
        .ROWS(ROWS),
        .COLS(COLS),
        .DATA_WIDTH(DATA_WIDTH),
        .IMAGE_ROWS(IMAGE_ROWS),
        .IMAGE_COLS(IMAGE_COLS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .image(image),
        .done(done)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 10ns clock period

    // File paths
    string input_image_file = "./src/files/golden_input.mem";
    string golden_output_file = "./src/files/golden_conv.mem";

    // Memory to hold the golden output
    logic signed [DATA_WIDTH-1:0] golden_output[KERNELS][IMAGE_ROWS-ROWS+1][IMAGE_COLS-COLS+1];

    // Load image data into the image array
    initial begin
        $readmemh(input_image_file, image);
        $display("Input image loaded successfully!");
    end

    // Load golden output data
    initial begin
        $readmemh(golden_output_file, golden_output);
        $display("Golden output loaded successfully!");
    end

    // Task to compare results with golden output
    task compare_results;
        integer k, r, c;
        integer mismatches;
        begin
            mismatches = 0;
            for (k = 0; k < KERNELS; k = k + 1) begin
                $display("Comparing results for Kernel %0d", k);
                for (r = 0; r < IMAGE_ROWS-ROWS+1; r = r + 1) begin
                    for (c = 0; c < IMAGE_COLS-COLS+1; c = c + 1) begin
                        if (dut.conv_result[k][r][c] !== golden_output[k][r][c]) begin
                            $display("Mismatch at Kernel %0d, Row %0d, Col %0d: DUT = %0d, Golden = %0d",
                                     k, r, c, dut.conv_result[k][r][c], golden_output[k][r][c]);
                            mismatches = mismatches + 1;
                        end else begin 
                        $display("MATCH at Kernel %0d, Row %0d, Col %0d: DUT = %0d, Golden = %0d",k, r, c, dut.conv_result[k][r][c], golden_output[k][r][c]);
                        end
                    end
                end
            end
            if (mismatches == 0) begin
                $display("All results match the golden output!");
            end else begin
                $display("Total mismatches: %0d", mismatches);
            end
        end
    endtask

    // Test sequence
    initial begin
        // Reset the module
        rst = 0;
        start = 0;
        #20 rst = 1;
        #20 rst = 0;
        #20 start = 1;

        // Wait for the 'done' flag from the DUT
        wait (dut.done);
        $display("Processing complete. Checking results...");

        // Compare results
        compare_results;

        $stop;
    end

    // Waveform generation
    initial begin
        $dumpfile("waveform.vcd"); // VCD file name
        $dumpvars(0, tb_convolution_reuse);
    end
endmodule

