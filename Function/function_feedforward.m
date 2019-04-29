function [var_error,var_sumlay1,var_sumlay2,var_lay1,var_lay2,var_trueclass]=function_feedforward...
    (var_w1,var_w2,var_count,var_countw,patterns_tr,d_tr,var_n,trueclass_tr)
    %Layer 1
%     var_sumlay1=[1;patterns_tr(:,var_count)]'*var_w1(:,:,var_countw);
    for var_i=1:var_n
        var_sumlay1(var_i)=var_w1(1,var_i,var_countw)...
        +patterns_tr(1,var_count)*var_w1(2,var_i,var_countw)...
        +patterns_tr(2,var_count)*var_w1(3,var_i,var_countw);

    end
    var_lay1 =tanh(var_sumlay1);
    %Layer 2
    var_sumlay2=[1,var_lay1             ]*var_w2(:,:,var_countw);
    var_lay2 =tanh(var_sumlay2);
    %Output error
    var_error  =d_tr(:,var_count)-var_lay2';
    var_trueclass  =isequal(find(var_lay2==max(var_lay2)),trueclass_tr(var_count));
end