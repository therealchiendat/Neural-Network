function [var_find]=function_outvar(var_w1,var_w2,var_count,var_countw,patterns_tr)
    
    %Layer 1
    var_sumlay1=sum([1;patterns_tr(:,var_count)].*var_w1(:,:,var_countw),1);
    var_lay1 =tanh(var_sumlay1);
    %Layer 2
    var_sumlay2=sum([1;var_lay1'             ].*var_w2(:,:,var_countw),1);
    var_lay2 =tanh(var_sumlay2);
    %Output
    var_find =find(var_lay2==max(var_lay2));    
end