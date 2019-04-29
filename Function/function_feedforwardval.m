function [var_error_val,var_trueclass,var_lay2]=...
    function_feedforwardval(var_w1,var_w2,var_count,var_countw,patterns_v,trueclass_v,d_v,var_n,var_sumlay1)
    
    %Layer 1
%     var_sumlay1=patterns_v(var_count,:)*var_w1(:,:,var_countw);
    for var_i=1:var_n
        var_sumlay1(var_i)=var_w1(1,var_i,var_countw)...
        +patterns_v(1,var_count)*var_w1(2,var_i,var_countw)...
        +patterns_v(2,var_count)*var_w1(3,var_i,var_countw);

    end
    var_lay1 =tanh(var_sumlay1);
    %Layer 2
    var_sumlay2=[1,var_lay1             ]*var_w2(:,:,var_countw);
    var_lay2 =tanh(var_sumlay2);
    %Output error
    d_v(trueclass_v(var_count))=1;
    var_error_val  =(d_v-var_lay2)';
    var_trueclass  =isequal(find(var_lay2==max(var_lay2)),trueclass_v(var_count));
end