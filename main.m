%% Artificial Neural Networks
%   Assignment 4, Question 3
%   Author: Chien Dat Nguyen Dinh , S/N: 3483629
%   
%  Disclaimer: This code has been tested to work with Matlab 2017a
%              it is not guarantee that it would work as expected on the
%              older version of matlab
%% Project Definition:
% * The fourth assignment of the course Artificial Neural Networks
%   (ECE6503) requested to apply a neural network to a pettern
%   classification problem for which the goal is to classify an object as
%   belonging to one of 3 classes based on 2 extracted features.

%% ______________THE CODE START AFTER THIS LINE________________


%% Clear
clear
clc
clf
close all
addpath('.\Function')

%% Small note:
%============
%   I will add from error handling every time the code tries to call to a
%   file outside of the script to prevent the code from stopping unexpectedly

%% Initialization
% Load the value from the data set:
    
    try
        load('.\Data\data_ass4_2017.mat');
    catch exception
        msgbox({'Make sure the file name "data_ass4_2017.mat" in the subfolder "..\Data"'},...
            'Cannot find data','error')
        return
    end

% Choose between 3 and 30 first-layer units:

    v_threeunits = imread('.\Pictures\3layer.jpg');
    v_thirtyunits = imread('.\Pictures\30layer.jpg');
    v_select = listdlg('PromptString','Select number of first layer units:',...
                    'SelectionMode','single','Name','First layer unit number:',...
                    'ListString',{'3','30'});
    v_choose=figure(1);
    set(v_choose,'Name','Sketch','NumberTitle','off','MenuBar','none',...
        'Color',[1 1 1])
    if v_select==1            
    imshow(rgb2gray(v_threeunits))
    title('Sketch of the 2-layered Neural Network (3 first-layer units)')
    v_n=3;
    else
    imshow(rgb2gray(v_thirtyunits))    
    title('Sketch of the 2-layered Neural Network (30 first-layer units)')
    v_n=30;
    end
    text(5,5,'f()=tanh()','FontSize',15)
    %% Main script area
    %----------
    %Defining variable
    v_trial=10;
    v_wtlim=0.1;
    rng(v_trial*2);
    v_w1 =v_wtlim*rand(3,v_n,1000)*2-0.1;  % A very small initial random weight value.
    rng(v_trial*1);
    v_w2 =v_wtlim*rand(v_n+1,3,1000)*2-0.1; % Also a very small random weight value.
        
    v_mu   =0.01;
    v_error=zeros(120,3);
    v_error_val=zeros(120,3);
    v_lay1 =zeros(120,v_n);
    v_lay2 =zeros(120,3);
    v_xi   =zeros(1,10000);
    v_xi_v=zeros(1,10000);
    v_xi_t =zeros(1,10000);
    v_classerror_v=zeros(1,10000);
    v_classerror=zeros(1,10000);
    %Update accumulator
    v_w1acc=zeros(3,v_n);
    v_w2acc=zeros(v_n+1,3);
    %Loading message
    v_waitbar=waitbar(0,'Training in process...');
    v_total   =10000;
    %Speedup variables:
    v_valpattern=[ones(120,1),patterns_v'];
    v_inputlay1=[1;0;0];
    v_inputlay2=ones(v_n+1,1);
    v_d=-1*ones(1,3);
    v_find_trueclass    =zeros(1,120);
    v_find_trueclass_val=zeros(1,120);
    v_find_trueclass_t  =zeros(1,120);
    
%% Start training:
%--------------------------------------------------------------------
    for v_epoch=1:v_total
        for v_cnt=1:120
            %Feed forward :
                %Training set
            [v_error(v_cnt,:),v_sumlay1,v_sumlay2,v_lay1(v_cnt,:),v_lay2(v_cnt,:),v_find_trueclass(v_cnt)]= ...
                function_feedforward(v_w1,v_w2,v_cnt,v_epoch,patterns_tr,d_tr,v_n,trueclass_tr);
                %Validation set
            [v_error_val(v_cnt,:),v_find_trueclass_val(v_cnt)]=...
                function_feedforwardval(v_w1,v_w2,v_cnt,v_epoch,patterns_v,trueclass_v,v_d,v_n,v_sumlay1);
                %Test set
            [v_error_t(v_cnt,:),v_find_trueclass_t(v_cnt)]=...
                function_feedforwardval(v_w1,v_w2,v_cnt,v_epoch,patterns_ts,trueclass_ts,v_d,v_n,v_sumlay1);
            %Feed back    :
            [v_deltalay1,v_deltalay2]=...
                function_feedback(v_w2,v_cnt,v_epoch,v_error,v_sumlay1,v_sumlay2,v_n);

            %Weight update accumulator:    
                for i=2:3
                    v_inputlay1(i)=patterns_tr(i-1,v_cnt);
                end
                for i=2:v_n+1
                   v_inputlay2(i)=v_lay1(v_cnt,i-1); 
                end
                for v_i=1:v_n
                    for v_j=1:3
                        v_w1acc(v_j,v_i)=v_w1acc(v_j,v_i)...
                        +v_mu*v_deltalay1(v_i)*v_inputlay1(v_j);
                    end
                end
                for v_i=1:3
                    for v_j=1:v_n+1
                        v_w2acc(v_j,v_i)=v_w2acc(v_j,v_i)...
                        +v_mu*v_deltalay2(v_i)*v_inputlay2(v_j);
                    end
                end
        end 
        %------------------------------------------------------------------
            %Weight and xi update update :
            v_w1(:,:,v_epoch+1)=v_w1(:,:,v_epoch)...
                +2/120*v_w1acc(:,:);
            v_w2(:,:,v_epoch+1)=v_w2(:,:,v_epoch)...
                +2/120*v_w2acc(:,:);
            v_xi(v_epoch)=mean(sum(v_error(1:120,:).^2,2));
            v_xi_v(v_epoch)=mean(sum(v_error_val(1:120,:).^2,2));
            v_xi_t(v_epoch)=mean(sum(v_error_t(1:120,:).^2,2));
            %Classification error rate update:
            v_classerror(v_epoch)=(120-sum(v_find_trueclass))/120*100;
            v_classerror_v(v_epoch)=(120-sum(v_find_trueclass_val))/120*100;
            v_classerror_t(v_epoch)=(120-sum(v_find_trueclass_t))/120*100;
            %avgSNO
            v_avgSNO(v_epoch)=mean(sum(v_lay2(1:120,:),2));
            %stdSNO
            v_stdSNO(v_epoch)=sqrt((mean(sum(v_lay2(1:120,:),2)-v_avgSNO(v_epoch)))^2);
            if v_epoch>1 
            if v_xi_v(v_epoch-1)<v_xi_v(v_epoch),flag=flag+1;
            else
                flag=0;
            end
            if abs(v_xi_v(v_epoch)-v_xi(v_epoch))>0.4
                flag1=flag1+1;
            else
                flag1=0;
            end
            if flag==500||flag1==20,waitbar(1),break,end
            if flag==500, waitbar(1),break,end
            if flag1==20, v_stop=v_epoch;end
            end
        %------------------------------------------------------------------
            v_w1acc(:,:)=0;
            v_w2acc(:,:)=0;
            if rem(v_epoch,v_total/100)==0
            waitbar(v_epoch/v_total);
            end
    end
    
    
    
    
    
    
%% Done training
%---------------------------------------------------------------------    
    
    close(v_waitbar)
    save('Output\Exporteddata.mat','v_w1','v_w2','v_n','v_epoch','v_trial')
    v_find     =zeros(1,120);
    [xq,yq]=meshgrid(linspace(-30,30,1000),linspace(-30,30,1000));
    
%% Plot result
%---------------------------------------------------------------------
%NOTE: The result will be plotted in tabs
%     v_epoch=6000
    close all
%     
    desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
    myGroup = desktop.addGroup('Training result');
    desktop.setGroupDocked('Training result', 0);
    myDim   = java.awt.Dimension(4, 1);   % 3 columns, 1 rows
    % 1: Maximized, 2: Tiled, 3: Floating
    desktop.setDocumentArrangement('Training result', 2, myDim)
    tabfigure    = gobjects(1, 4);
    bakWarn = warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
    
    %__________
    tabfigure(1)=figure('WindowStyle', 'docked', ...
        'Name', 'Map Training', 'NumberTitle', 'off',...
        'MenuBar','none','Color',[1 1 1]);
    set(get(handle(tabfigure(1)), 'javaframe'), 'GroupName', 'Training result');
        %----First Tab-----------
        v_plpattern=patterns_tr;
        v_maps=[patterns_tr,patterns_v,patterns_ts];
        for v_cnt=1:360
            [v_find(v_cnt)]=function_outvar(v_w1,v_w2,v_cnt,v_epoch,v_maps);
        end
        
        vq=griddata(v_maps(1,:),v_maps(2,:),v_find,xq,yq,'nearest');
        surf(xq,yq,vq,'edgecolor','none')
        shading interp
        v_colormap=[0.5 0.5 0.5;0.7 0.7 0.7;0.9 0.9 0.9];
        colormap(v_colormap)
        view(0,90)
        hold on
        plot3(v_plpattern(1,1:40)  ,v_plpattern(2,1:40)  ,3*ones(1,40),'bo',...
              v_plpattern(1,41:80) ,v_plpattern(2,41:80) ,3*ones(1,40),'bx',...
              v_plpattern(1,81:120),v_plpattern(2,81:120),3*ones(1,40),'bs')
          for i=1:120
              if trueclass_tr(i)~=v_find(i)
                  locpat(i)=1;
              end
          end
          loc_tr=find(locpat);
          for i=1:length(loc_tr)
              if loc_tr(i)<40
                  plot3(patterns_tr(1,loc_tr(i)),patterns_tr(2,loc_tr(i)),3,'ro')
              elseif loc_tr(i)>80
                  plot3(patterns_tr(1,loc_tr(i)),patterns_tr(2,loc_tr(i)),3,'rs')
              else
                  plot3(patterns_tr(1,loc_tr(i)),patterns_tr(2,loc_tr(i)),3,'rx')
              end
          end
          
            %Labeling
            xlabel('x_1');
            ylabel('x_2');
            title({'Illustration of input space get mapped by the neural network'...
                ,['Pattern: Training, \mu=',num2str(v_mu),' , wtlim=',num2str(v_wtlim)],...
                ['Epoch when training is done:', num2str(v_epoch)]})
        %----End First Tab----------
        
        tabfigure(2)=figure('WindowStyle', 'docked', ...
        'Name', 'Map Testing', 'NumberTitle', 'off',...
        'MenuBar','none','Color',[1 1 1]);
    set(get(handle(tabfigure(2)), 'javaframe'), 'GroupName', 'Training result');
        %----Second Tab-----------
        v_plpattern=patterns_ts;
        surf(xq,yq,vq,'edgecolor','none')
        shading interp
        v_colormap=[0.5 0.5 0.5;0.7 0.7 0.7;0.9 0.9 0.9];
        colormap(v_colormap)
        view(0,90)
        hold on
        plot3(v_plpattern(1,1:40)  ,v_plpattern(2,1:40)  ,3*ones(1,40),'bo',...
              v_plpattern(1,41:80) ,v_plpattern(2,41:80) ,3*ones(1,40),'bx',...
              v_plpattern(1,81:120),v_plpattern(2,81:120),3*ones(1,40),'bs')
          
          
          for i=1:120
              if trueclass_ts(i)~=v_find(i+240)
                  locpat1(i)=1;
              end
          end
          loc_ts=find(locpat1);
          for i=1:length(loc_ts)
              if loc_ts(i)<40
                  plot3(patterns_ts(1,loc_ts(i)),patterns_ts(2,loc_ts(i)),3,'ro')
              elseif loc_ts(i)>80
                  plot3(patterns_ts(1,loc_ts(i)),patterns_ts(2,loc_ts(i)),3,'rs')
              else
                  plot3(patterns_ts(1,loc_ts(i)),patterns_ts(2,loc_ts(i)),3,'rx')
              end
          end
            %Labeling
            xlabel('x_1');
            ylabel('x_2');
            title({'Illustration of input space get mapped by the neural network'...
                ,['Pattern: Testing, \mu=',num2str(v_mu),' , wtlim=',num2str(v_wtlim)],...
                ['Epoch when training is done:', num2str(v_epoch)]})
        %----End Second Tab----------

    tabfigure(3)=figure('WindowStyle', 'docked', ...
        'Name', 'Map Validation', 'NumberTitle', 'off',...
        'MenuBar','none','Color',[1 1 1]);
    set(get(handle(tabfigure(3)), 'javaframe'), 'GroupName', 'Training result');
        %----Third Tab-----------
        v_plpattern=patterns_v;

        surf(xq,yq,vq,'edgecolor','none')
        shading interp
        v_colormap=[0.5 0.5 0.5;0.7 0.7 0.7;0.9 0.9 0.9];
        colormap(v_colormap)
        view(0,90)
        hold on
        plot3(v_plpattern(1,1:40)  ,v_plpattern(2,1:40)  ,3*ones(1,40),'bo',...
              v_plpattern(1,41:80) ,v_plpattern(2,41:80) ,3*ones(1,40),'bx',...
              v_plpattern(1,81:120),v_plpattern(2,81:120),3*ones(1,40),'bs')
          
          
          for i=1:120
              if trueclass_v(i)~=v_find(i+120)
                  locpat2(i)=1;
              end
          end
          loc_v=find(locpat2);
          for i=1:length(loc_v)
              if loc_v(i)<40
                  plot3(patterns_v(1,loc_v(i)),patterns_v(2,loc_v(i)),3,'ro')
              elseif loc_v(i)>80
                  plot3(patterns_v(1,loc_v(i)),patterns_v(2,loc_v(i)),3,'rs')
              else
                  plot3(patterns_v(1,loc_v(i)),patterns_v(2,loc_v(i)),3,'rx')
              end
          end
            %Labeling
            xlabel('x_1');
            ylabel('x_2');
            title({'Illustration of input space get mapped by the neural network'...
                ,['Pattern: Validation, \mu=',num2str(v_mu),' , wtlim=',num2str(v_wtlim)],...
                ['Epoch when training is done:', num2str(v_epoch)]})
        %----End Third Tab----------
    %__________    
    tabfigure(4)=figure('WindowStyle', 'docked', ...
        'Name', 'ASSE vs time', 'NumberTitle', 'off',...
        'MenuBar','none','Color',[1 1 1]);
    set(get(handle(tabfigure(4)), 'javaframe'), 'GroupName', 'Training result');
        %----Fourth Tab-----------
        v_xi=v_xi(1:v_epoch);
        v_xi_v=v_xi_v(1:v_epoch);
        plot_tr=semilogx(v_xi,'b'); 
        hold on
        plot_v =semilogx(v_xi_v,'r');
            %Labeling
            title({'\xi_{TR} and \xi_{V} over time',...
                ['Epoch number at stop= ', num2str(v_epoch),' \mu=',num2str(v_mu),' , wtlim=',num2str(v_wtlim)]...
                ,['\xi_{TR}= ',num2str(v_xi(v_epoch)),' ,\xi_{V}= ', num2str(v_xi_v(v_epoch))]});
            xlabel('training time (Total epoch)')
            ylabel('\xi')
            zoom on
%             text(v_stop,v_xi(v_stop)+0.05,['\downarrow Should stop training at epoch=',num2str(v_stop)])
            uicontrol('Style', 'text', 'String', 'Select part of the graph to zoom, double click to zoom out',...
                'Position', [35 20 400 15],'BackgroundColor',[1 1 1])
            legend('\xi_{TR}','\xi_{VS}')
            %Interaction
            btn = uicontrol('Style', 'togglebutton', 'String', 'VS CURVE',...
                'Position', [20 20 60 20],...
                'Callback', {@callback_plot_en,plot_v});
            btn1 = uicontrol('Style', 'togglebutton', 'String', 'TR CURVE',...
                'Position', [20 50 60 20],...
                'Callback', {@callback_plot_en,plot_tr});
        %----End Fourth Tab----------

    %__________    
    tabfigure(5)=figure('WindowStyle', 'docked', ...
        'Name', 'Classerror vs time', 'NumberTitle', 'off',...
        'MenuBar','none','Color',[1 1 1]);
    set(get(handle(tabfigure(5)), 'javaframe'), 'GroupName', 'Training result');
        %----Fifth Tab-----------
        v_classerror=v_classerror(1:v_epoch);
        v_classerror_v=v_classerror_v(1:v_epoch);
        plot_cetr=semilogx(v_classerror,'b'); 
        hold on
        plot_cev =semilogx(v_classerror_v,'r');
            %Labeling
            title({'Classification error_{TR} and Classification error_{V} over time',...
                ['Epoch number at stop= ', num2str(v_epoch),' \mu=',num2str(v_mu),' , wtlim=',num2str(v_wtlim)],...
                ['Classification error_{TR}= ',num2str(v_classerror(v_epoch)),' ,Classification error_{V}= ', num2str(v_classerror_v(v_epoch))]});
            xlabel('training time (Total epoch)')
            ylabel('Percentage (%)')
            zoom on
            uicontrol('Style', 'text', 'String', 'Select part of the graph to zoom, double click to zoom out',...
                'Position', [35 20 400 15],'BackgroundColor',[1 1 1])
            legend('Classification error_{TR}','Classification error_{VS}')
            %Interaction
            btn = uicontrol('Style', 'togglebutton', 'String', 'VS CURVE',...
                'Position', [20 20 60 20],...
                'Callback', {@callback_plot_en,plot_cev});
            btn1 = uicontrol('Style', 'togglebutton', 'String', 'TR CURVE',...
                'Position', [20 50 60 20],...
                'Callback', {@callback_plot_en,plot_cetr});
        %----End Fifth Tab----------

    desktop.setDocumentArrangement('Training result', 1, myDim)
    warning(bakWarn);
    