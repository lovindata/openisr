import NavBarOrg from "@/v2/features/shared/components/organisms/NavBarOrg";
import { ModalProvider } from "@/v2/features/shared/hooks/contexts/Modal";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function MainLayoutTpl({ children }: Props) {
  return (
    <ModalProvider>
      <div className="md:flex md:flex-row">
        <NavBarOrg />
        <div className="h-full w-full">{children}</div>
      </div>
    </ModalProvider>
  );
}
