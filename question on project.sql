SELECT * FROM titanic.titanic;

/* age of  older male with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where sex='male' and survived='1' ;

/* age of  older female with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where sex='female' and survived='1' ;
/* age of  younger female and male with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where sex='female' and survived='1' 
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where sex='male' and survived='1' ;





/*ist class younger female and male with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where pclass='1' and survived='1'  and sex ='male'
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where pclass='1' and survived='1'and sex ='female' 
/*2st class younger female and male with survived */
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where pclass='2' and survived='1' and sex ='male'
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where pclass='2' and survived='1'  and  sex ='female'
union
/*3st class younger female and male with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where pclass='3' and survived='1'  and  sex ='female'
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,min(age) from titanic  where pclass='3' and survived='1' and sex ='male';


/*ist class older female and male with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where pclass='1' and survived='1'  and sex ='male'
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where pclass='1' and survived='1'and sex ='female' 
/*2st class younger female and male with survived */
union
select PassengerId,pclass,sex,name,ticket,fare,embarkedmax(age) from titanic  where pclass='2' and survived='1' and sex ='male'
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where pclass='2' and survived='1'  and  sex ='female'
union
/*3st class younger female and male with survived */
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where pclass='3' and survived='1'  and  sex ='female'
union
select PassengerId,pclass,sex,name,ticket,fare,embarked,max(age) from titanic  where pclass='3' and survived='1' and sex ='male';





/* about ticket 
max and min ticket

*/
select PassengerId,pclass,sex,name,max(ticket),fare,embarked,max(age) from titanic  where  pclass='3' and survived='0' and sex ='male';
select PassengerId,pclass,sex,name,max(ticket),fare,embarked,age from titanic order by embarked;






/* find the person*/
select PassengerId,pclass,sex,name,ticket,fare,embarked,age from titanic  where PassengerId='Leonardo DiCaprio';

select PassengerId,pclass,sex,name,ticket,fare,embarked,age from titanic  where PassengerId='Kate Winslet';
select PassengerId,pclass,sex,name,ticket,fare,embarked,age from titanic  where PassengerId='Kathy Bates';




/* describe of data*/
select pclass,sex,name,avg(PassengerId)  from titanic
union
select pclass,sex,name,min(age) from titanic;

select pclass,sex,name,avg(age)  from titanic where pclass='1'
union
select pclass,sex,name,avg(ticket)  from titanic;


select  pclass,sex,name , avg(Survived) from titanic order by Ticket desc;

