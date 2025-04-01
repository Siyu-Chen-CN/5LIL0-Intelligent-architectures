module string_match_tb;

    logic clk;
    logic rst;
    logic [95:0] search_string;
    logic found;
    logic signed [31:0] position;

    // TO INSTANTIATE THE MODULE
    string_match uut (
        .clk(clk),
        .rst(rst),
        .search_string(search_string),
        .found(found),
        .position(position)
    );

    // Clock generation
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst = 1;
        #20;
        rst = 0;
        // Test input string (concatenation of three 32-bit words)
        search_string = 96'b11111111111111111111111111101110_11111111111111111111111111110100_11111111111111111111111111111100;

        // TO DO COMPLETE 
        #50;
        
        // Display results
        if (found)
            $display("Match found at position: %d (in 32-bit words)", position);
        else
            $display("Match not found.");
        $finish;
    end
endmodule