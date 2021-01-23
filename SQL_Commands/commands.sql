UNIQUE:
--------
create table test(id integer,name varchar(20),unique);
desc table;
create unique index idxl on test(id);
desc test;
alter table test drop index idzl;
desc test;
#VIEWS: window on data;
---------------------------
select * from myemp limit 5;
create view empview as select first_name,last_name,hire_date,salary from emp;
select * from empview limit 5;
create view mydept as select * from myemp where dep_id =50;
select * from mydept;
#
create view aview as select * from authors where authorid < 6;
select * from aview;
insert into aview values(9,'Shaw');#inserted thorugh view#
select * from authors;
create view aview2 as select * from authors where authorid < 6 with check option;
insert into aview2 values(11,'Bleach');#check option failed#


select * from mmovies;
select * from members;
create view moview as select title,first_name,last_name from movies inner join members on members.movieid = movies.id;
select * from moview;
create view myview as select distinct dep_id from myemp;
select * from myview;
#stored procedures:
-----------------------

