      character*320 input
      character*1 chain_id, peptide_chain_id
      character*4 atname
      character*10 resnum_target,resnum_start,resnum_end
      character*80 line
      real ctarget(3),clig(3,100)
      call getarg(1,input)
      call getarg(2,chain_id)
      call getarg(3,resnum_target)
      read(resnum_target,*) iresnum_target
      call getarg(4,resnum_start)
      read(resnum_start,*) iresnum_start
      call getarg(5,resnum_end)
      read(resnum_end,*) iresnum_end
      call getarg(6,peptide_chain_id)
      open(1,file=input)
      i=0
      do i=1,3
      ctarget(i)=0.0
      enddo
      do
      read(1,'(a80)',end=10) line
      if (line(1:4).ne."ATOM") goto 100
      read (line(12:16),*) atname
      if (atname.eq."CA") then
          if ((line(22:22).eq.chain_id)) then
              read (line(23:26),*) ires
              if ((ires.eq.iresnum_target).and.(ctarget(1).eq.0.0)) then
              read(line(31:54),'(3f8.3)') (ctarget(j),j=1,3)
              endif
          endif
          read (line(23:26),*) ires
          if (ires.eq.iresnum_start) then
              do i=iresnum_start,iresnum_end
                  do j=1,3
                      clig(j,i)=0.0
                  enddo
              enddo
          endif
          if ((ires.ge.iresnum_start).and.(ires.le.iresnum_end)) then
              if (line(22:22).eq.peptide_chain_id) then
              read(line(31:54),'(3f8.3)') (clig(j,ires),j=1,3)
              endif
          endif
      endif
 100  continue
      enddo
 10   continue
      imax=min(ires,iresnum_end)
      rmeandist=0.0
      do i=iresnum_start,imax
      dist=0.0
      do j=1,3
      dist=dist+(clig(j,i)-ctarget(j))**2
      enddo
      rmeandist=rmeandist+sqrt(dist)
      enddo
      write(2,*) rmeandist/(imax-iresnum_start+1)
      end
