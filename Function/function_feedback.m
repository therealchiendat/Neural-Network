function [var_deltalay1,var_deltalay2]=function_feedback...
    (var_w2,var_count,var_countw,var_error,var_sumlay1,var_sumlay2,var_n)
    
    %Layer 2
    var_deltalay2=(1-(tanh(var_sumlay2)).^2).*var_error(var_count,:);
    var_sum   =sum(var_deltalay2.*var_w2(2:var_n+1,:,var_countw),2);
%     var_sum      =sum(var_outsum,2);
    %Layer 1
    var_deltalay1=(1-(tanh(var_sumlay1)).^2).*var_sum';

end


